// Copyright: 2020 Simon A. Nielsen Knights
// License: ISC

const std = @import("std");
const Blake3 = std.crypto.Blake3;
const b64 = std.base64.standard_encoder;
const fnv = std.hash.Fnv1a_64;

const TagType = enum { topic, medium, author, license, isbn, doi, language };

const Token = union(enum) {
    TagType: struct { len: usize, tag: TagType },
    Tag: usize,
    Title: usize,
    Description: usize,
    Hash: usize,
    Link: usize,

    pub fn slice(self: Token, text: []const u8, index: usize) []const u8 {
        switch (self) {
            .Tag, .Title, .Hash, .Link => |x| return text[1 + index - x .. index],
            .Description => |x| return text[1 + index - x .. index - 1],
            .TagType => |x| return text[1 + index - x.len .. index],
        }
    }
};

const StreamingParser = struct {
    count: usize,
    state: State,
    hash: fnv,

    const State = enum {
        LinkEnd,
        TagType,
        Tag,
        TagEnd,
        Title,
        Desc,
        DescCont,
        Hash,
        HashEnd,
        Link,
    };

    pub fn init() StreamingParser {
        var p: StreamingParser = undefined;
        p.reset();
        return p;
    }

    pub fn reset(self: *StreamingParser) void {
        self.count = 0;
        self.state = .LinkEnd;
        self.hash = fnv.init();
    }

    fn getTag(hash: u64) ?TagType {
        inline for (@typeInfo(TagType).Enum.fields) |field| {
            if (hash == comptime fnv.hash(field.name)) {
                return std.meta.stringToEnum(TagType, field.name);
            }
        }
        return null;
    }

    pub fn feed(self: *StreamingParser, c: u8) !?Token {
        self.count += 1;
        switch (self.state) {
            .LinkEnd => switch (c) {
                '#' => {
                    self.count = 0;
                    self.state = .TagType;
                },
                else => return error.InvalidSection,
            },
            .TagType => switch (c) {
                '\n' => return error.MissingTagBody,
                ':' => {
                    const token = Token{
                        .TagType = .{
                            .len = self.count,
                            .tag = getTag(self.hash.final()) orelse return error.InvalidTag,
                        },
                    };
                    self.hash = fnv.init();
                    self.state = .Tag;
                    self.count = 0;
                    return token;
                },
                else => {
                    self.hash.update(([1]u8{c})[0..]);
                },
            },
            .Tag => switch (c) {
                '\n' => {
                    const token = Token{ .Tag = self.count };
                    self.state = .TagEnd;
                    return token;
                },
                else => {},
            },
            .TagEnd => switch (c) {
                '#' => {
                    self.count = 0;
                    self.state = .TagType;
                },
                else => {
                    self.count = 1;
                    self.state = .Title;
                },
            },
            .Title => switch (c) {
                '\n' => {
                    const token = Token{ .Title = self.count };
                    self.count = 0;
                    self.state = .Desc;
                    return token;
                },
                else => {},
            },
            .Desc => switch (c) {
                '\n' => self.state = .DescCont,
                else => {},
            },
            .DescCont => switch (c) {
                '^' => {
                    const token = Token{ .Description = self.count };
                    self.count = 0;
                    self.state = .Hash;
                    return token;
                },
                '@' => return error.InvalidSection,
                '#' => return error.InvalidSection,
                else => self.state = .Desc,
            },
            .Hash => switch (c) {
                '\n' => {
                    self.state = .HashEnd;
                    return Token{ .Hash = self.count };
                },
                else => {},
            },
            .HashEnd => switch (c) {
                '@' => {
                    self.count = 0;
                    self.state = .Link;
                },
                else => return error.InvalidSection,
            },
            .Link => switch (c) {
                '\n' => {
                    self.state = .LinkEnd;
                    return Token{ .Link = self.count };
                },
                else => {},
            },
        }
        return null;
    }
};

const TokenStream = struct {
    sp: StreamingParser,
    text: []const u8,
    index: usize,

    pub fn init(text: []const u8) TokenStream {
        var p: TokenParser = undefined;
        p.reset();
        p.text = text;
        return p;
    }

    pub fn next(self: *StreamingParser) void {
        self.sp = StreamingParser.init();
        self.index = 0;
    }

    pub fn next(self: *StreamingParser) !?Token {}
};

test "" {
    const resources = @embedFile("../res");
    var p = StreamingParser.init();

    for (resources) |byte, i| {
        if (try p.feed(byte)) |item| {}
    }
}

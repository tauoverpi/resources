// Copyright: 2020 Simon A. Nielsen Knights
// License: ISC

const std = @import("std");
const Blake3 = std.crypto.Blake3;
const b64 = std.base64.standard_encoder;
const fnv = std.hash.Fnv1a_64;

const TagType = enum { Topic, Medium, Author, License };

const Token = union(enum) {
    TagType: struct { len: usize, tag: TagType },
    Tag: usize,
    Title: usize,
    Description: usize,
    Hash: usize,
    Link: usize,

    pub fn slice(self: Token, index: usize, text: []const u8) []const u8 {
        switch (self) {
            .Tag, .Title, .Description, .Hash, .Link => |x| return text[index .. index - x],
            .TagType => |x| return text[index .. index - x.len],
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
                            // TODO: move this out of the parser
                            .tag = switch (self.hash.final()) {
                                fnv.hash("topic") => .Topic,
                                fnv.hash("medium") => .Medium,
                                fnv.hash("author") => .Author,
                                fnv.hash("license") => .License,
                                else => return error.InvalidTagType,
                            },
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

test "" {
    const resources = @embedFile("../res");
    var p = StreamingParser.init();

    for (resources) |byte| {
        if (try p.feed(byte)) |item| {
            std.debug.warn("{}\n", .{item});
        }
    }
}

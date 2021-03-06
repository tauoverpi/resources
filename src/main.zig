// Copyright: 2020 Simon A. Nielsen Knights
// License: ISC

const std = @import("std");
const Blake3 = std.crypto.Blake3;
const b64 = std.base64.standard_encoder;
const fnv = std.hash.Fnv1a_64;
const TagMap = std.AutoHashMap([]const u8, std.SegmentedList([]const u8, 2));

const TagType = enum { topic, author, medium, license, isbn, doi, language };

// required tags for a successful parse
const Required = struct {
    language: bool,
    medium: bool,
    topic: bool,
};

comptime {
    for (std.meta.fields(Required)) |field| {
        if (!@hasField(TagType, field.name)) {
            @compileError("required field doesn't exist");
        }
    }
}

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
    required: Required,

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
        inline for (std.meta.fields(Required)) |field| {
            @field(self.required, field.name) = false;
        }
    }

    fn getTag(self: *StreamingParser, hash: u64) !TagType {
        inline for (std.meta.fields(TagType)) |field| {
            if (hash == comptime fnv.hash(field.name)) {
                if (@hasField(Required, field.name)) {
                    @field(self.required, field.name) = true;
                }
                return comptime @intToEnum(TagType, field.value);
            }
        }
        return error.InvalidTag;
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
                            .tag = try getTag(self, self.hash.final()),
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
                    inline for (std.meta.fields(Required)) |field| {
                        defer @field(self.required, field.name) = false;
                        if (!@field(self.required, field.name)) {
                            return error.MissingRequiredField;
                        }
                    }
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

    pub fn reset(self: *StreamingParser) void {
        self.sp = StreamingParser.init();
        self.index = 0;
    }

    pub fn next(self: *StreamingParser) !?Token {}
};

test "" {
    const resources = @embedFile("../res");
    var p = StreamingParser.init();
    var line: usize = 0;

    for (resources) |byte, i| {
        if (byte == '\n') line += 1;
        if (p.feed(byte) catch |e| {
            switch (e) {
                error.MissingRequiredField => std.debug.print("missing required on line {}\n", .{line}),
                else => {},
            }
            return e;
        }) |item| {}
    }
}

fn html(title: []const u8, desc: []const u8, link: []const u8, out_stream: anytype) void {
    out_stream.writeAll("<div class=\"resource\">");
    out_stream.writeAll("  <div class=\"resource-title\">");
    out_stream.writeAll("</div");
}

fn markdown(title: []const u8, desc: []const u8, link: []const u8, out_stream: anytype) void {
    out_stream.writeAll("[");
    out_stream.writeAll(title);
    out_stream.writeAll("](");
    out_stream.writeAll(link);
    out_stream.writeAll(")\n> ");
    out_stream.writeAll(desc);
    out_stream.writeAll("\n");
}

// Copyright: 2020 Simon A. Nielsen Knights
// License: ISC

const std = @import("std");
const Blake3 = std.crypto.Blake3;

const Token = union(enum) {
    Tag: usize,
    Title: usize,
    Description: usize,
    Hash: usize,
    Link: usize,

    pub fn slice(self: Token, index: usize, text: []const u8) usize {
        switch (self) {
            .Tag, .Title, .Description, .Hash, .Link => |x| return text[0..],
        }
    }
};

const StreamingParser = struct {
    count: usize,
    state: State,

    const State = enum {
        LinkEnd,
        Tag,
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
    }

    pub fn feed(self: *StreamingParser, c: u8) !?Token {
        self.count += 1;
        switch (self.state) {
            .LinkEnd => switch (c) {
                '#' => {
                    self.count = 0;
                    self.state = .Tag;
                },
                else => return error.InvalidSection,
            },
            .Tag => switch (c) {
                '\n' => {
                    const token = Token{ .Tag = self.count };
                    self.count = 0;
                    self.state = .Title;
                    return token;
                },
                else => {},
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
    const resources =
        \\#thing
        \\stuff
        \\described
        \\^hashed
        \\@linked
        \\#morething
        \\#withtags
        \\and a tile with spaces!
        \\then a very short description
        \\which spans more than one line
        \\as it must be able to handle it
        \\^that unused hash
        \\@oh noes, a link
        \\
    ;

    var p = StreamingParser.init();

    for (resources) |byte| {
        if (try p.feed(byte)) |item| {
            _ = item.slice(9, resources[0..]);
            std.debug.warn("{}\n", .{item});
        }
    }
}

const Resource = struct {
    topics: TopicIterator,
};

pub fn main() anyerror!void {
    std.debug.warn("All your codebase are belong to us.\n", .{});
}

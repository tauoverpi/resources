# A List of Resources

This project collects a list of resources in a graph structure and collects
tools to work with it.

The format:

```
#topic:any string following is a valid topic until the next line break
#topic:there can be multiple topics per item
#author:authors have their own field
#author:which can have multiple entries
#medium:can currently be multiple
Title covers a single line
While the description may span multiple lines to accommodate full abstracts
rather than just a short note on the resource. All text is encoded as utf-8 with
unix-style line endings (only '\n') and may not start a line with any of the
reserved characters #^@ where ^ first on a line signifies the end of the
description and the beginning of the resource checksum.
^a+checksum+of+the+resource+follows+in+base64+using+blake3==
@https://then.finally.the/resource/itself/as/a/url.pdf
```

## TODO

- EBNF
- the rest of this list

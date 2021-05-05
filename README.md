# About

// TODO

# Warnings

I see ... on my console, what does it mean?

### Undefined type encountered while encoding symbols

Sometimes volatility symbols define structs with pointers to other structs it has no definition for. This should not
be a showstopper, but hampers the accuracy of the model. NoSysms treats these pointers the same way it treats void
pointers: We know that these bytes are supposed to be pointers, but it doesn't "follow" them while encoding.

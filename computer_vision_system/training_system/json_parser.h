#ifndef jsonparcer_H
#define jsonparcer_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef jsonparcer_TYPE_U64

#include <stdint.h>

typedef uint64_t jsonparcer_u64;
#endif

#ifndef jsonparcer_TYPE_S64

#include <stdint.h>

typedef uint64_t jsonparcer_s64;
#endif

typedef enum json_parcer_type {
  json_parcer_NULL,    // this is null value
  json_parcer_OBJECT,  // this is an object; properties can be found in child
                       // nodes
  json_parcer_ARRAY,   // this is an array; items can be found in child nodes
  json_parcer_STRING,  // this is a string; value can be found in text_value
                       // field
  json_parcer_INTEGER, // this is an integer; value can be found in int_value
                       // field
  json_parcer_DOUBLE, // this is a double; value can be found in dbl_value field
  json_parcer_BOOL // this is a boolean; value can be found in int_value field
} json_parcer_type;

typedef struct json_parcer {
  json_parcer_type type; // type of json node, see above
  const char *key;       // key of the property; for object's children only
  union {
    const char *text_value; // text value of STRING node
    struct {
      union {
        jsonparcer_u64 u_value; // the value of INTEGER or BOOL node
        jsonparcer_s64 s_value;
      };
      double dbl_value; // the value of DOUBLE node
    } num;
    struct { // children of OBJECT or ARRAY
      int length;
      struct json_parcer *first;
      struct json_parcer *last;
    } children;
  };
  struct json_parcer *next; // points to next child
} json_parcer;

typedef int (*json_parcer_unicode_encoder)(unsigned int codepoint, char *p,
                                           char **endp);

extern json_parcer_unicode_encoder json_parcer_unicode_to_utf8;

const json_parcer *json_parcer_parse(char *text,
                                     json_parcer_unicode_encoder encoder);

const json_parcer *json_parcer_parse_utf8(char *text);

void json_parcer_free(const json_parcer *js);

const json_parcer *
json_parcer_get(const json_parcer *json,
                const char *key); // get object's property by key
const json_parcer *json_parcer_item(const json_parcer *json,
                                    int idx); // get array element by index

#ifdef __cplusplus
}
#endif

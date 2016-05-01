#ifndef UTIL_H

#define UTIL_H


#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
    /* (x, y) */                                                                    \
    (p0) = tilted + (rect).x + (step) * (rect).y,                                   \
    /* (x - h, y + h) */                                                            \
    (p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
    /* (x + w, y + w) */                                                            \
    (p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
    /* (x + w - h, y + w + h) */                                                    \
    (p3) = tilted + (rect).x + (rect).width - (rect).height                         \
           + (step) * ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
    ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

#define CV_SUM_OFS( p0, p1, p2, p3, sum, rect, step )                 \
/* (x, y) */                                                          \
(p0) = sum + (rect).x + (step) * (rect).y,                            \
/* (x + w, y) */                                                      \
(p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
/* (x + w, y) */                                                      \
(p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
/* (x + w, y + h) */                                                  \
(p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_OFS( p0, p1, p2, p3, tilted, rect, step )                     \
/* (x, y) */                                                                    \
(p0) = tilted + (rect).x + (step) * (rect).y,                                   \
/* (x - h, y + h) */                                                            \
(p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
/* (x + w, y + w) */                                                            \
(p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
/* (x + w - h, y + w + h) */                                                    \
(p3) = tilted + (rect).x + (rect).width - (rect).height                         \
+ (step) * ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

#define CALC_SUM_OFS_(p0, p1, p2, p3, ptr) \
((ptr)[p0] - (ptr)[p1] - (ptr)[p2] + (ptr)[p3])

#define CALC_SUM_OFS(rect, ptr) CALC_SUM_OFS_((rect)[0], (rect)[1], (rect)[2], (rect)[3], ptr)







#endif /* end of include guard: UTIL_H */

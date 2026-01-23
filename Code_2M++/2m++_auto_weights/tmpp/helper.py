import numpy as np

__all__=['add_field']

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError, "`A' must be a structured numpy array"
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


def grow_catalog(catalog1, catalog2, marker_name=None, marker1=0, marker2=1):
    
    if (catalog1.dtype != catalog2.dtype):
        raise ValueError("dtypes must be identical")
    
    c0 = np.empty(catalog1.shape[0]+catalog2.shape[0],dtype=catalog1.dtype)
    
    c0[0:catalog1.shape[0]] = catalog1
    c0[catalog1.shape[0]:] = catalog2
    
    if (marker_name != None):
        try:
            v = c0[marker_name][0]
        except ValueError:
            c0 = add_field(c0, [(marker_name, 'i')])
            
        c0[marker_name][0:catalog1.shape[0]] = marker1
        c0[marker_name][catalog1.shape[0]:] = marker2
            
    return c0

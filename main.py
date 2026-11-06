try:
    from IPython.display import display
except ImportError:
    display = print



dataiku.Dataset("LDS_MPS_TRD_HARIC_MAIL").write_with_schema(long)

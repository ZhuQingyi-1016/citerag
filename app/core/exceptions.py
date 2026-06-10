class AppError(Exception):
    """Base class for application-level errors."""
    pass


class FileRepoError(AppError):
    """Base class for file repository errors."""
    pass


class FileNotFoundError(FileRepoError):
    """Raised when a file_id cannot be found in storage."""
    pass


class DuplicateFileIdError(FileRepoError):
    """Raised when multiple files match the same file_id."""
    pass


class IndexErrorBase(AppError):
    """Base class for indexing/retrieval errors."""
    pass


class UnsupportedFileTypeError(IndexErrorBase):
    """Raised when a file type is not supported for indexing."""
    pass


class FileNotIndexedError(IndexErrorBase):
    """Raised when a specific file has not been indexed yet."""
    pass


class NoIndexedFilesError(IndexErrorBase):
    """Raised when global search is requested but no files are indexed."""
    pass
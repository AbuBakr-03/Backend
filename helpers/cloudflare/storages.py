from storages.backends.s3 import S3Storage


class StaticFileStorage(S3Storage):
    # helpers.cloudflare.storages.StaticFileStorage
    location = "static"


class MediaFileStorage(S3Storage):
    # helpers.cloudflare.storages.MediaFileStorage
    location = "media"
    file_overwrite = False  # This prevents overwrites

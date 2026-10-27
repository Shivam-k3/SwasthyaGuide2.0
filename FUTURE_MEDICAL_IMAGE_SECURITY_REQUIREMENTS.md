# FUTURE_MEDICAL_IMAGE_SECURITY_REQUIREMENTS.md
**Date:** 2026-10-02 12:57:08

## Security Requirements Before Adding Medical Image Uploads

**DO NOT implement image upload in this security stage.** If adding pneumonia/breast cancer imaging later:

Note: this API has **no upload path today**. The request-body size limit in
`app/security.py` (1 MiB, 413) applies to JSON only and is not an image-upload
control; do not treat it as one. Item 3 below is the upload-specific limit and
would need to coexist with it.

1. MIME allowlist (e.g., image/png, image/jpeg, image/dicom where applicable)
2. Actual file signature/content verification (not just extension/MIME)
3. Max file size (bytes)
4. Max image dimensions (width×height)
5. Max decompressed size (to prevent zip bombs)
6. Malformed image rejection
7. Resource exhaustion protection (timeouts, memory limits)
8. Temporary isolated storage (no shared/public dirs)
9. No public file serving
10. No permanent storage by default
11. Cleanup after inference (guaranteed)
12. Cleanup on exceptions (finally blocks)
13. EXIF/metadata handling (strip or ignore PHI)
14. No image contents in logs
15. No PHI in filenames/logs
16. Rate limiting (strict)
17. Request timeout
18. Model inference resource limits
19. Content-Type validation + magic bytes
20. Safe error handling

**Never trust:** filename, extension, or browser MIME type alone.

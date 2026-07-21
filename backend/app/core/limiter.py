"""
app/core/limiter.py
Shared slowapi Limiter instance.

Kept in its own module (rather than defined in main.py) so route modules
(register.py, verify.py) can import it directly to decorate their
endpoints with @limiter.limit(...) — slowapi's actual supported API —
without a circular import against main.py.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

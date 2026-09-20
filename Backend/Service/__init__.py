"""Service layer: modules that sit between controllers and the outside world.

Controllers own HTTP. Services own everything a controller should not have
to know — which vendor, which key, how to retry, what it cost.
"""

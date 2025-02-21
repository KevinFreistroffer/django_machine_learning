from django import template

register = template.Library()

@register.filter
def subtract(value, arg):
    """Subtract the arg from the value."""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return ''

@register.filter
def absolute(value):
    """Return the absolute value."""
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return '' 
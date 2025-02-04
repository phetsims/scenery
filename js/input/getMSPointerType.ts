// Copyright 2025, University of Colorado Boulder

/**
 * Returns the pointer type (for the pointer spec) for a given MS pointer event.
 *
 * (scenery-internal)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export const getMSPointerType = ( event: PointerEvent ): string => {
  // @ts-expect-error -- legacy API
  if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_TOUCH ) {
    return 'touch';
  }
  // @ts-expect-error -- legacy API
  else if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_PEN ) {
    return 'pen';
  }
  // @ts-expect-error -- legacy API
  else if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_MOUSE ) {
    return 'mouse';
  }
  else {
    return event.pointerType; // hope for the best
  }
};
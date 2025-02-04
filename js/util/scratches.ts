// Copyright 2025, University of Colorado Boulder

/*
 * Holds elements that can be reused from multiple places for temporary uses.
 * This is useful for performance reasons, as it avoids creating and destroying
 * elements repeatedly.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';

// NOTE: may have arbitrary state
export const scratchCanvas = document.createElement( 'canvas' );
scenery.register( 'scratchCanvas', scratchCanvas );

// NOTE: may have arbitrary state
export const scratchContext = scratchCanvas.getContext( '2d' )!;
scenery.register( 'scratchContext', scratchContext );
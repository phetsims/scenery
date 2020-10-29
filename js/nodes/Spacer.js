// Copyright 2015-2020, University of Colorado Boulder

/**
 * A Node meant to just take up certain bounds. It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import scenery from '../scenery.js';
import Leaf from './Leaf.js';
import Node from './Node.js';

class Spacer extends Node {
  /**
   * Creates a spacer taking up a rectangular area from x: [0,width] and y: [0,height]. Use x/y in options to control
   * its position.
   * @public
   *
   * @param {number} width - The width of the spacer
   * @param {number} height - The height of the spacer
   * @param {Object} [options] - Passed to Node
   */
  constructor( width, height, options ) {
    assert && assert( typeof width === 'number' && isFinite( width ), 'width should be a finite number' );
    assert && assert( typeof height === 'number' && isFinite( height ), 'height should be a finite number' );

    super();

    // override the local bounds to our area
    this.localBounds = new Bounds2( 0, 0, width, height );

    this.mutate( options );
  }
}

Leaf.mixInto( Spacer ); // prevent children from being added, since we're overriding local bounds

scenery.register( 'Spacer', Spacer );
export default Spacer;
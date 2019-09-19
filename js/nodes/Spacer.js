// Copyright 2015-2019, University of Colorado Boulder

/**
 * A Node meant to just take up certain bounds. It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const Bounds2 = require( 'DOT/Bounds2' );
  const inherit = require( 'PHET_CORE/inherit' );
  const Leaf = require( 'SCENERY/nodes/Leaf' );
  const Node = require( 'SCENERY/nodes/Node' );
  const scenery = require( 'SCENERY/scenery' );

  /**
   * Creates a spacer taking up a rectangular area from x: [0,width] and y: [0,height]. Use x/y in options to control
   * its position.
   * @public
   * @constructor
   * @extends Node
   *
   * @param {number} width - The width of the spacer
   * @param {number} height - The height of the spacer
   * @param {Object} [options] - Passed to Node
   */
  function Spacer( width, height, options ) {
    assert && assert( typeof width === 'number' && isFinite( width ), 'width should be a finite number' );
    assert && assert( typeof height === 'number' && isFinite( height ), 'height should be a finite number' );

    Node.call( this );

    // override the local bounds to our area
    this.localBounds = new Bounds2( 0, 0, width, height );

    this.mutate( options );
  }

  scenery.register( 'Spacer', Spacer );

  inherit( Node, Spacer );
  Leaf.mixInto( Spacer ); // prevent children from being added, since we're overriding local bounds

  return Spacer;
} );

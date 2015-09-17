//  Copyright 2002-2015, University of Colorado Boulder

/**
 * Flexible input listener for a node, based on a composite/chain of responsibility/mixin type pattern.
 * See https://github.com/phetsims/scenery/issues/465
 *
 * TODO: A better name?
 *
 * TODO: This file is highly volatile and not ready for public consumption.  It is being actively developed
 * as part of https://github.com/phetsims/scenery-phet/issues/186 for work in Bending Light.  If it cannot be generalized
 * for usage in other simulations, it will be moved to Bending Light.  The API and implementation are subject to
 * change, and this disclaimer will be removed when the code is ready for review or usage.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   *
   * @constructor
   */
  function RelativeDragHandler( node ) {
    this.node = node;
  }

  scenery.RelativeDragHandler = RelativeDragHandler;

  return inherit( Object, RelativeDragHandler, {
    start: function( event, trail, chainInputListener, point ) {
      this.startOffset = event.currentTarget.globalToParentPoint( point ).minus( this.node.translation );
      chainInputListener.nextStart( event, trail, point );
    },
    drag: function( event, trail, chainInputListener, point ) {
      point = event.currentTarget.globalToParentPoint( point ).minus( this.startOffset );
      chainInputListener.nextDrag( event, trail, point );
    },
    end: function( event, trail, chainInputListener, point ) {
      this.startOffset = null;
      chainInputListener.nextEnd( event, trail, point );
    }
  } );
} );
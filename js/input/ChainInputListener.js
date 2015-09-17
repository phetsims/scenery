// Copyright 2002-2014, University of Colorado Boulder

/**
 * Flexible input listener for a node, based on a composite/chain of responsibility/mixin type pattern.
 * See https://github.com/phetsims/scenery/issues/465
 *
 * TODO: A better name?
 *
 * TODO: How about a linked list for the traversal, instead of calling back to the parent ChainInputListener?
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

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SimpleDragHandler = require( 'SCENERY/input/SimpleDragHandler' );

  /**
   *
   * TODO: Should this list be mutable?
   * @param {Handler[]} handlers - array of handlers that will process the data
   * @param {Object} [options]
   *
   * @constructor
   */
  scenery.ChainInputListener = function ChainInputListener( handlers, options ) {
    var chainInputListener = this;
    this.handlers = handlers;
    options = _.extend( {}, options );

    this.currentHandler = 0;

    // TODO: Cannot we generalize this pattern to handle original scenery events, and work for all input types, not just 
    // TODO: mouse/touch/drag on the pointer?
    options.start = function( event, trail ) {
      chainInputListener.currentHandler = 0;
      chainInputListener.nextStart( event, trail );
    };
    options.drag = function( event, trail ) {
      chainInputListener.currentHandler = 0;
      chainInputListener.nextDrag( event, trail );
    };
    options.end = function( event, trail ) {
      chainInputListener.currentHandler = 0;
      chainInputListener.nextEnd( event, trail );
    };

    SimpleDragHandler.call( this, options );
  };

  var ChainInputListener = scenery.ChainInputListener;

  return inherit( SimpleDragHandler, ChainInputListener, {
    nextStart: function( event, trail ) {
      var nextHandler = this.handlers[ this.currentHandler++ ];
      nextHandler && nextHandler.start( event, trail, this );
    },
    nextDrag: function( event, trail ) {
      var nextHandler = this.handlers[ this.currentHandler++ ];
      nextHandler && nextHandler.drag( event, trail, this );
    },
    nextEnd: function( event, trail ) {
      var nextHandler = this.handlers[ this.currentHandler++ ];
      nextHandler && nextHandler.end( event, trail, this );
    }
  } );
} );



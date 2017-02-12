// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var protoAfter = require( 'SCENERY/listeners/protoAfter' );

  function TouchSnaggableConstructor( options ) {
    options = _.extend( {
      allowTouchSnag: false // TODO: decide on appropriate default
    }, options );

    this._allowTouchSnag = options.allowTouchSnag;
  }

  var TouchSnaggable = {
    createMixin: function( predicate ) {
      assert && assert( typeof predicate === 'function' );

      return function TouchSnaggable( pressListenerType ) {
        var proto = pressListenerType.prototype;

        proto.mixesTouchSnaggable = true;

        // TODO: note about overriding? seems unclean
        proto.tryTouchSnag = function tryToSnag( event ) {
          if ( predicate.call( this, event ) ) {
            this.tryPress( event );
          }
        };

        [ 'touchenter', 'touchmove' ].forEach( function( eventName ) {
          protoAfter( proto, eventName, proto.tryTouchSnag );
        } );

        return TouchSnaggableConstructor;
      };
    }
  };

  TouchSnaggable.mixin = TouchSnaggable.createMixin( function() {
    return this._allowTouchSnag;
  } );

  scenery.register( 'TouchSnaggable', TouchSnaggable );

  return TouchSnaggable;
} );

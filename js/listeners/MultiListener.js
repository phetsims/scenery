// Copyright 2013-2017, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   *
   * @params {Object} [options] - See the constructor body (below) for documented options.
   */
  function MultiListener( options ) {
    var self = this;

    options = _.extend( {
      mouseButton: 0, // TODO: see PressListener

      pressCursor: 'pointer', // TODO: see PressListener
    }, options );

    this._mouseButton = options.mouseButton;
    this._pressCursor = options.pressCursor;

    // @private {Array.<Press>}
    this._presses = [];

    // @private
    this._pressListener = {
      move: function( event ) {
        self.movePress( self.findPress( event.pointer ) );
      },

      up: function( event ) {
        // TODO: consider logging press on the pointer itself?
        self.removePress( self.findPress( event.pointer ) );
      },

      cancel: function( event ) {
        var press = self.findPress( event.pointer );
        press.interrupted = true;

        self.removePress( press );
      }
    };
  }

  scenery.register( 'MultiListener', MultiListener );

  inherit( Object, MultiListener, {

    findPress: function( pointer ) {
      for ( var i = 0; i < this._presses.length; i++ ) {
        if ( this._presses[ i ].pointer === pointer ) {
          return this._presses[ i ];
        }
      }
      assert && assert( false, 'Did not find press' );
      return null;
    },

    // TODO: see PressListener
    down: function( event ) {
      this.tryPress( event );
    },

    // TODO: see PressListener
    tryPress: function( event ) {
      if ( this.isPressed ) { return; }

      if ( event.pointer.isMouse && event.domEvent.button !== this._mouseButton ) { return; }

      this.addPress( new Press( event.pointer ) );
    },

    addPress: function( press ) {
      this._presses.push( press );

      press.pointer.cursor = this._pressCursor;
      press.pointer.addListener( this._pressListener );

      // TODO: handle interrupted
    },

    movePress: function( press ) {

    },

    removePress: function( press ) {
      press.pointer.removeListener( this._pressListener );
      press.pointer.cursor = null;

      arrayRemove( this._presses, press );
    }


  } );

  function Press( pointer ) {
    this.pointer = pointer;
    this.interrupted = false;
  }

  inherit( Object, Press, {
    // ..?
  } );

  return MultiListener;
} );

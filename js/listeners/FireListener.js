// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: name (because ButtonListener was taken)
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Property = require( 'AXON/Property' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );

  /**
   * TODO: doc
   */
  function FireListener( options ) {
    var self = this;

    options = _.extend( {
      isOverProperty: new Property( false ), // allowing this to be overridden helps with button models
      fireOnDown: false,
      fire: null, // TODO: fire can't take an event right now, because release doesn't
      isHighlightedProperty: new Property( false ) // allowing this to be overridden helps with button models
    }, options );

    PressListener.call( this, options );

    this.overCountProperty = new Property( 0 );
    this.isOverProperty = options.isOverProperty;

    this.overCountProperty.link( function( overCount ) {
      self.isOverProperty.value = overCount > 0;
    } );

    this._fireOnDown = options.fireOnDown;
    this._fireCallback = options.fire;

    this.isHighlightedProperty = options.isHighlightedProperty;

    // TODO: Don't highlight for "pressed" pointers that weren't pressed for us
    Property.multilink( [ this.isOverProperty, this.isPressedProperty ], function( isOver, isPressed ) {
      self.isHighlightedProperty.value = isOver || isPressed;
    } );

    // Hoverable, Fireable, Highlightable, TouchSnaggable
  }

  scenery.register( 'FireListener', FireListener );

  inherit( PressListener, FireListener, {
    enter: function( event ) {
      this.overCountProperty.value++;
    },

    exit: function( event ) {
      assert && assert( this.overCountProperty.value >= 0, 'Exit event not matched by an enter event' );

      this.overCountProperty.value--;
    },

    fire: function() {
      this._fireCallback && this._fireCallback();
    },

    // @override
    press: function( event ) {
      PressListener.prototype.press.call( this, event );

      if ( this._fireOnDown ) {
        this.fire( event );
      }
    },

    // @override
    release: function() {
      PressListener.prototype.release.call( this, event );

      if ( !this._fireOnDown && this.isOverProperty.value && !this.wasInterrupted ) {
        this.fire();
      }
    }
  } );

  return FireListener;
} );

// Copyright 2013-2017, University of Colorado Boulder

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

    // TODO: highlight should ignore attached pointers that isn't our pointer that's pressing us
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
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener enter' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.overCountProperty.value++;

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    exit: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener exit' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( this.overCountProperty.value >= 0, 'Exit event not matched by an enter event' );

      this.overCountProperty.value--;

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    fire: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener fire' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this._fireCallback && this._fireCallback();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    // @override
    press: function( event ) {
      var success = PressListener.prototype.press.call( this, event );

      if ( success ) {
        if ( this._fireOnDown ) {
          this.fire( event );
        }
      }

      return success;
    },

    // @override
    release: function() {
      PressListener.prototype.release.call( this, event );

      if ( !this._fireOnDown && this.isOverProperty.value && !this.interrupted ) {
        this.fire();
      }
    }
  } );

  return FireListener;
} );

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
  var ObservableArray = require( 'AXON/ObservableArray' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );

  /**
   * TODO: doc
   */
  function FireListener( options ) {
    var self = this;

    options = _.extend( {
      isHoveringProperty: new Property( false ), // allowing this to be overridden helps with button models
      fireOnDown: false,
      fire: null, // TODO: fire can't take an event right now, because release doesn't
      isHighlightedProperty: new Property( false ) // allowing this to be overridden helps with button models
    }, options );

    PressListener.call( this, options );

    this.overPointers = new ObservableArray();
    this.overCountProperty = this.overPointers.lengthProperty;
    assert && assert( this.overCountProperty instanceof Property && this.overCountProperty.value === 0 );
    this.isHoveringProperty = options.isHoveringProperty;

    var isHoveringListener = this.invalidateHover.bind( this );
    this.overCountProperty.link( isHoveringListener );
    this.isPressedProperty.link( isHoveringListener );
    this.overPointers.addItemAddedListener( function( pointer ) {
      pointer.isDownProperty.link( isHoveringListener );
    } );
    this.overPointers.addItemRemovedListener( function( pointer ) {
      pointer.isDownProperty.unlink( isHoveringListener );
    } );

    this._fireOnDown = options.fireOnDown;
    this._fireCallback = options.fire;

    this.isHighlightedProperty = options.isHighlightedProperty;

    Property.multilink( [ this.isHoveringProperty, this.isPressedProperty ], function( isHovering, isPressed ) {
      self.isHighlightedProperty.value = isHovering || isPressed;
    } );
  }

  scenery.register( 'FireListener', FireListener );

  inherit( PressListener, FireListener, {
    invalidateHover: function() {
      var pointers = this.overPointers.getArray();
      for ( var i = 0; i < pointers.length; i++ ) {
        var pointer = pointers[ i ];
        if ( !pointer.isDown || pointer === this.pointer ) {
          this.isHoveringProperty.value = true;
          return;
        }
      }
      this.isHoveringProperty.value = false;
    },

    enter: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener enter' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.overPointers.push( event.pointer );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    exit: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener exit' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( this.overPointers.contains( event.pointer ), 'Exit event not matched by an enter event' );

      this.overPointers.remove( event.pointer );

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

      if ( !this._fireOnDown && this.isHoveringProperty.value && !this.interrupted ) {
        this.fire();
      }
    },

    /**
     * Disposes the listener, releasing references. It should not be used after this.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener dispose' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // We need to release references to any pointers that are over us.
      this.overPointers.clear();

      PressListener.prototype.dispose.call( this );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  } );

  return FireListener;
} );

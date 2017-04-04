// Copyright 2017, University of Colorado Boulder

/**
 * A listener for common button usage, providing the fire() method/callback and helpful properties.
 *
 * TODO: name (because ButtonListener was taken). Can we rename the old ButtonListener and have this be ButtonListener?
 *
 * TODO: unit tests
 *
 * TODO: add example usage
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
   * @constructor
   * @extends PressListener
   *
   * @param {Object} [options] - See the constructor body (below) and in PressListener for documented options.
   */
  function FireListener( options ) {
    options = _.extend( {
      // {Function|null} - Called as fire() when the button is fired.
      fire: null,

      // {boolean} - If true, the button will fire when the button is pressed. If false, the button will fire when the
      // button is released while the pointer is over the button.
      fireOnDown: false,

      // {Property.<boolean>} - A property that will be controlled by this listener. It will be set to true when at
      // least one pointer is over the button.
      // A custom property may be passed in here, as it may be useful for hooking up to existing button models.
      isOverProperty: new Property( false ),

      // {Property.<boolean>} - A property that will be controlled by this listener. It will be set to true when either:
      //   1. The button is pressed and the pointer that is pressing is over the button.
      //   2. There is at least one unpressed pointer that is over the button.
      // A custom property may be passed in here, as it may be useful for hooking up to existing button models.
      isHoveringProperty: new Property( false ),

      // {Property.<boolean>} - A property that will be controlled by this listener. It will be set to true when either:
      //   1. The button is pressed.
      //   2. There is at least one unpressed pointer that is over the button.
      // This is essentially true when ( isPressed || isHovering ).
      // A custom property may be passed in here, as it may be useful for hooking up to existing button models.
      isHighlightedProperty: new Property( false )
    }, options );

    assert && assert( options.fire === null || typeof options.fire === 'function',
      'The fire callback, if provided, should be a function' );
    assert && assert( typeof options.fireOnDown === 'boolean', 'fireOnDown should be a boolean' );
    assert && assert( options.isOverProperty instanceof Property && options.isOverProperty.value === false,
      'If a custom isOverProperty is provided, it must be a Property that is false initially' );
    assert && assert( options.isHoveringProperty instanceof Property && options.isHoveringProperty.value === false,
      'If a custom isHoveringProperty is provided, it must be a Property that is false initially' );
    assert && assert( options.isHighlightedProperty instanceof Property && options.isHighlightedProperty.value === false,
      'If a custom isHighlightedProperty is provided, it must be a Property that is false initially' );

    PressListener.call( this, options );

    // @public {ObservableArray.<Pointer>} - Contains all pointers that are over our button. Tracked by adding with
    // 'enter' events and removing with 'exit' events.
    this.overPointers = new ObservableArray();

    // @public {Property.<boolean>} - See options for documentation.
    this.isOverProperty = options.isOverProperty;
    this.isHoveringProperty = options.isHoveringProperty;
    this.isHighlightedProperty = options.isHighlightedProperty;

    // @private - See options for documentation.
    this._fireOnDown = options.fireOnDown;
    this._fireCallback = options.fire;


    // isOverProperty updates (not a DerivedProperty because we need to hook to passed-in properties)
    this.overPointers.lengthProperty.link( this.invalidateOver.bind( this ) );

    // isHoveringProperty updates (not a DerivedProperty because we need to hook to passed-in properties)
    var isHoveringListener = this.invalidateHovering.bind( this );
    this.overPointers.lengthProperty.link( isHoveringListener );
    this.isPressedProperty.link( isHoveringListener );
    // overPointers will be cleared on disposal, so we should release references then.
    this.overPointers.addItemAddedListener( function( pointer ) {
      pointer.isDownProperty.link( isHoveringListener );
    } );
    this.overPointers.addItemRemovedListener( function( pointer ) {
      pointer.isDownProperty.unlink( isHoveringListener );
    } );

    // isHighlightedProperty updates (not a DerivedProperty because we need to hook to passed-in properties)
    var isHighlightedListener = this.invalidateHighlighted.bind( this );
    this.isHoveringProperty.link( isHighlightedListener );
    this.isPressedProperty.link( isHighlightedListener );
  }

  scenery.register( 'FireListener', FireListener );

  inherit( PressListener, FireListener, {

    /**
     * Fires any associated button fire callback.
     * @public
     *
     * NOTE: This is safe to call on the listener externally.
     */
    fire: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener fire' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this._fireCallback && this._fireCallback();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'enter' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    enter: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener enter' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.overPointers.push( event.pointer );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'exit' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    exit: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener exit' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( this.overPointers.contains( event.pointer ), 'Exit event not matched by an enter event' );

      this.overPointers.remove( event.pointer );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Recomputes the value for isOverProperty. Separate to reduce anonymous function closures.
     * @private
     */
    invalidateOver: function() {
      this.isOverProperty.value = this.overPointers.length > 0;
    },

    /**
     * Recomputes the value for isHoveringProperty. Separate to reduce anonymous function closures.
     * @private
     */
    invalidateHovering: function() {
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

    /**
     * Recomputes the value for isHighlightedProperty. Separate to reduce anonymous function closures.
     * @private
     */
    invalidateHighlighted: function() {
      this.isHighlightedProperty.value = this.isHoveringProperty.value || this.isPressedProperty.value;
    },

    /**
     * Presses the button.
     * @public
     * @override
     *
     * NOTE: This is safe to call externally in order to attempt to start a press. fireListener.canPress( event ) can
     * be used to determine whether this will actually start a press.
     *
     * @param {Event} event
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press: function( event ) {
      var success = PressListener.prototype.press.call( this, event );

      if ( success ) {
        if ( this._fireOnDown ) {
          this.fire( event );
        }
      }

      return success;
    },

    /**
     * Releases the button.
     * @public
     * @override
     *
     * NOTE: This can be safely called externally in order to force a release of this button (no actual 'up' event is
     * needed). If the cancel/interrupt behavior is more preferable (will not fire the button), then call interrupt()
     * on this listener instead.
     */
    release: function() {
      PressListener.prototype.release.call( this );

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      // TODO: Is this a problem that we can't access things like this.pointer here?
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

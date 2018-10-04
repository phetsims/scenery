// Copyright 2017, University of Colorado Boulder

/**
 * A listener for common button usage, providing the fire() method/callback and helpful properties.
 *
 * For example usage, see scenery/examples/input.html. Usually you can just pass a fire callback and things work.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Emitter = require( 'AXON/Emitter' );
  var inherit = require( 'PHET_CORE/inherit' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );
  var scenery = require( 'SCENERY/scenery' );
  var Tandem = require( 'TANDEM/Tandem' );

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

      tandem: Tandem.optional
    }, options );

    assert && assert( options.fire === null || typeof options.fire === 'function',
      'The fire callback, if provided, should be a function' );
    assert && assert( typeof options.fireOnDown === 'boolean', 'fireOnDown should be a boolean' );

    PressListener.call( this, options );

    // @private - See options for documentation.
    this._fireOnDown = options.fireOnDown;

    // @private - for PhET-iO events
    this.firedEmitter = new Emitter( {
      tandem: options.tandem.createTandem( 'firedEmitter' ),
      phetioEventType: 'user'
    } );
    options.fire && this.firedEmitter.addListener( options.fire );
  }

  scenery.register( 'FireListener', FireListener );

  inherit( PressListener, FireListener, {

    /**
     * Fires any associated button fire callback.
     * @public
     *
     * @param {Event|null} event
     * NOTE: This is safe to call on the listener externally.
     */
    fire: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener fire' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.firedEmitter.emit();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
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
      var self = this;

      var success = PressListener.prototype.press.call( this, event, undefined, function( success ) {
        // This function is only called on success
        if ( self._fireOnDown ) {
          self.fire( event );
        }
      } );
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
      var self = this;
      PressListener.prototype.release.call( this, function() {

        // Notify after the rest of release is called in order to prevent it from triggering interrupt().
        if ( !self._fireOnDown && self.isHoveringProperty.value && !self.interrupted ) {
          self.fire();
        }
      } );
    }
  } );

  return FireListener;
} );

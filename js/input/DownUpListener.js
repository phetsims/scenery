// Copyright 2013-2016, University of Colorado Boulder


/**
 * Basic down/up pointer handling for a Node, so that it's easy to handle buttons
 *
 * TODO: test hand handle down, go off screen, up. How to handle that properly?
 * TODO: tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  var Input = require( 'SCENERY/input/Input' );
  var Trail = require( 'SCENERY/util/Trail' );

  /*
   * The 'trail' parameter passed to down/upInside/upOutside will end with the node to which this DownUpListener has been added.
   *
   * Allowed options: {
   *    mouseButton: 0  // The mouse button to use: left: 0, middle: 1, right: 2, see https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent
   *    down: null      // down( event, trail ) is called when the pointer is pressed down on this node
   *                    // (and another pointer is not already down on it).
   *    up: null        // up( event, trail ) is called after 'down', regardless of the pointer's current location.
   *                    // Additionally, it is called AFTER upInside or upOutside, whichever is relevant
   *    upInside: null  // upInside( event, trail ) is called after 'down', when the pointer is released inside
   *                    // this node (it or a descendant is the top pickable node under the pointer)
   *    upOutside: null // upOutside( event, trail ) is called after 'down', when the pointer is released outside
   *                    // this node (it or a descendant is the not top pickable node under the pointer, even if the
   *                    // same instance is still directly under the pointer)
   * }
   */
  function DownUpListener( options ) {
    var self = this;

    options = _.extend( {
      mouseButton: 0 // allow a different mouse button
    }, options );
    this.options = options; // @private
    this.isDown = false;   // public, whether this listener is down
    this.downCurrentTarget = null; // 'up' is handled via a pointer lister, which will have null currentTarget, so save the 'down' currentTarget
    this.downTrail = null;
    this.pointer = null;
    this.interrupted = false;

    // this listener gets added to the pointer on a 'down'
    this.downListener = {
      // mouse/touch up
      up: function( event ) {
        sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'DownUpListener (pointer) up for ' + self.downTrail.toString() );
        assert && assert( event.pointer === self.pointer );
        if ( !event.pointer.isMouse || event.domEvent.button === self.options.mouseButton ) {
          self.buttonUp( event );
        }
      },

      interrupt: function() {
        self.interrupt();
      },

      // touch cancel
      cancel: function( event ) {
        sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'DownUpListener (pointer) cancel for ' + self.downTrail.toString() );
        assert && assert( event.pointer === self.pointer );
        self.buttonUp( event );
      },

      // When the enter or space key is released, trigger an up event
      // TODO: Only trigger this if the enter/space key went down for this node
      keyup: function( event ) {
        var keyCode = event.domEvent.keyCode;
        if ( keyCode === Input.KEY_ENTER || keyCode === Input.KEY_SPACE ) {
          self.buttonUp( event );
        }
      }
    };
  }

  scenery.register( 'DownUpListener', DownUpListener );

  inherit( Object, DownUpListener, {
    buttonDown: function( event ) {
      // already down from another pointer, don't do anything
      if ( this.isDown ) { return; }

      // ignore other mouse buttons
      if ( event.pointer.isMouse && event.domEvent.button !== this.options.mouseButton ) { return; }

      // add our listener so we catch the up wherever we are
      event.pointer.addInputListener( this.downListener );

      this.isDown = true;
      this.downCurrentTarget = event.currentTarget;
      this.downTrail = event.trail.subtrailTo( event.currentTarget, false );
      this.pointer = event.pointer;

      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'DownUpListener buttonDown for ' + this.downTrail.toString() );
      if ( this.options.down ) {
        this.options.down( event, this.downTrail );
      }
    },

    buttonUp: function( event ) {
      this.isDown = false;
      this.pointer.removeInputListener( this.downListener );

      var currentTargetSave = event.currentTarget;
      event.currentTarget = this.downCurrentTarget; // up is handled by a pointer listener, so currentTarget would be null.
      if ( this.options.upInside || this.options.upOutside ) {
        var trailUnderPointer = event.trail;

        // TODO: consider changing this so that it just does a hit check and ignores anything in front?
        var isInside = trailUnderPointer.isExtensionOf( this.downTrail, true ) && !this.interrupted;

        if ( isInside && this.options.upInside ) {
          this.options.upInside( event, this.downTrail );
        }
        else if ( !isInside && this.options.upOutside ) {
          this.options.upOutside( event, this.downTrail );
        }
      }
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'DownUpListener buttonUp for ' + this.downTrail.toString() );
      if ( this.options.up ) {
        this.options.up( event, this.downTrail );
      }
      event.currentTarget = currentTargetSave; // be polite to other listeners, restore currentTarget
    },

    /*---------------------------------------------------------------------------*
     * events called from the node input listener
     *----------------------------------------------------------------------------*/

    // mouse/touch down on this node
    down: function( event ) {
      this.buttonDown( event );
    },

    // Called when input is interrupted on this listener, see https://github.com/phetsims/scenery/issues/218
    interrupt: function() {
      if ( this.isDown ) {
        this.interrupted = true;

        // We create a synthetic event here, as there is no available event here.
        this.buttonUp( {
          // Empty trail, so that it for-sure isn't under our downTrail (guaranteeing that isInside will be false).
          trail: new Trail(),
          currentTarget: this.downCurrentTarget,
          pointer: this.pointer
        } );

        this.interrupted = false;
      }
    },

    // When enter/space pressed for this node, trigger a button down
    keydown: function( event ) {
      var keyCode = event.domEvent.keyCode;
      if ( keyCode === Input.KEY_ENTER || keyCode === Input.KEY_SPACE ) {
        this.buttonDown( event );
      }
    }
  } );

  return DownUpListener;
} );

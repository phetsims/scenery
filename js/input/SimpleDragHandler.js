// Copyright 2013-2016, University of Colorado Boulder


/**
 * Basic dragging for a node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /*
   * Allowed options: {
   *    allowTouchSnag: false // allow touch swipes across an object to pick it up. If a function is passed, the value allowTouchSnag( event ) is used
   *    dragCursor: 'pointer' // while dragging with the mouse, sets the cursor to this value (or use null to not override the cursor while dragging)
   *    mouseButton: 0        // allow changing the mouse button that activates the drag listener. -1 should activate on any mouse button, 0 on left, 1 for middle, 2 for right, etc.
   *    start: null           // if non-null, called when a drag is started. start( event, trail )
   *    drag: null            // if non-null, called when the user moves something with a drag (not a start or end event).
   *                                                                         drag( event, trail )
   *    end: null             // if non-null, called when a drag is ended.   end( event, trail )
   *    translate:            // if this exists, translate( { delta: _, oldPosition: _, position: _ } ) will be called.
   * }
   */
  function SimpleDragHandler( options ) {
    var self = this;

    options = _.extend( {
      allowTouchSnag: false,
      mouseButton: 0,
      dragCursor: 'pointer'
    }, options );
    this.options = options; // @private

    this.dragging = false;            // whether a node is being dragged with this handler
    this.pointer = null;              // the pointer doing the current dragging
    this.trail = null;                // stores the path to the node that is being dragged
    this.transform = null;            // transform of the trail to our node (but not including our node, so we can prepend the deltas)
    this.node = null;                 // the node that we are handling the drag for
    this.lastDragPoint = null;        // the location of the drag at the previous event (so we can calculate a delta)
    this.startTransformMatrix = null; // the node's transform at the start of the drag, so we can reset on a touch cancel
    this.mouseButton = undefined;     // tracks which mouse button was pressed, so we can handle that specifically
    this.interrupted = false;         // whether the last input was interrupted (available during endDrag)
    // TODO: consider mouse buttons as separate pointers?

    // if an ancestor is transformed, pin our node
    this.transformListener = {
      transform: function( args ) {
        if ( !self.trail.isExtensionOf( args.trail, true ) ) {
          return;
        }

        var newMatrix = args.trail.getMatrix();
        var oldMatrix = self.transform.getMatrix();

        // if A was the trail's old transform, B is the trail's new transform, we need to apply (B^-1 A) to our node
        self.node.prependMatrix( newMatrix.inverted().timesMatrix( oldMatrix ) );

        // store the new matrix so we can do deltas using it now
        self.transform.setMatrix( newMatrix );
      }
    };

    // this listener gets added to the pointer when it starts dragging our node
    this.dragListener = {
      // mouse/touch up
      up: function( event ) {
        if ( !self.dragging ) { return; }

        assert && assert( event.pointer === self.pointer, 'Wrong pointer in up' );
        if ( !event.pointer.isMouse || event.domEvent.button === self.mouseButton ) {
          var saveCurrentTarget = event.currentTarget;
          event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          self.endDrag( event );
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }
      },

      // touch cancel
      cancel: function( event ) {
        if ( !self.dragging ) { return; }

        assert && assert( event.pointer === self.pointer, 'Wrong pointer in cancel' );

        var saveCurrentTarget = event.currentTarget;
        event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
        self.endDrag( event );
        event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget

        // since it's a cancel event, go back!
        if ( !self.transform ) {
          self.node.setMatrix( self.startTransformMatrix );
        }
      },

      // mouse/touch move
      move: function( event ) {
        if ( !self.dragging ) { return; }

        assert && assert( event.pointer === self.pointer, 'Wrong pointer in move' );

        var globalDelta = self.pointer.point.minus( self.lastDragPoint );

        // ignore move events that have 0-length (Chrome seems to be auto-firing these on Windows, see https://code.google.com/p/chromium/issues/detail?id=327114)
        if ( globalDelta.magnitudeSquared() === 0 ) {
          return;
        }

        var delta = self.transform.inverseDelta2( globalDelta );

        // move by the delta between the previous point, using the precomputed transform
        // prepend the translation on the node, so we can ignore whatever other transform state the node has
        if ( self.options.translate ) {
          var translation = self.node.getMatrix().getTranslation();
          self.options.translate.call( null, {
            delta: delta,
            oldPosition: translation,
            position: translation.plus( delta )
          } );
        }
        self.lastDragPoint = self.pointer.point;

        if ( self.options.drag ) {
          // TODO: consider adding in a delta to the listener
          // TODO: add the position in to the listener
          var saveCurrentTarget = event.currentTarget;
          event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          self.options.drag.call( null, event, self.trail ); // new position (old position?) delta
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }
      },

      // pointer interruption
      interrupt: () => {
        self.interrupt();
      }
    };
  }

  scenery.register( 'SimpleDragHandler', SimpleDragHandler );

  inherit( Object, SimpleDragHandler, {
    startDrag: function( event ) {
      if ( this.dragging ) { return; }

      // set a flag on the pointer so it won't pick up other nodes
      event.pointer.dragging = true;
      event.pointer.cursor = this.options.dragCursor;
      event.pointer.addInputListener( this.dragListener );

      // set all of our persistent information
      this.dragging = true;
      this.pointer = event.pointer;
      this.trail = event.trail.subtrailTo( event.currentTarget, true );
      this.transform = this.trail.getTransform();
      this.node = event.currentTarget;
      this.lastDragPoint = event.pointer.point;
      this.startTransformMatrix = event.currentTarget.getMatrix().copy();
      // event.domEvent may not exist if this is touch-to-snag
      this.mouseButton = event.pointer.isMouse ? event.domEvent.button : undefined;

      if ( this.options.start ) {
        this.options.start.call( null, event, this.trail );
      }
    },

    endDrag: function( event ) {
      if ( !this.dragging ) { return; }

      this.pointer.dragging = false;
      this.pointer.cursor = null;
      this.pointer.removeInputListener( this.dragListener );
      this.dragging = false;

      if ( this.options.end ) {
        this.options.end.call( null, event, this.trail );
      }

      // release our reference
      this.pointer = null;
    },

    // Called when input is interrupted on this listener, see https://github.com/phetsims/scenery/issues/218
    interrupt: function() {
      if ( this.dragging ) {
        this.interrupted = true;

        // We create a synthetic event here, as there is no available event here.
        this.endDrag( {
          pointer: this.pointer,
          currentTarget: this.node
        } );

        this.interrupted = false;
      }
    },

    tryToSnag: function( event ) {
      // don't allow drag attempts that use the wrong mouse button (-1 indicates any mouse button works)
      if ( event.pointer.isMouse && event.domEvent && this.options.mouseButton !== event.domEvent.button && this.options.mouseButton !== -1 ) {
        return;
      }

      // only start dragging if the pointer isn't dragging anything, we aren't being dragged, and if it's a mouse it's button is down
      if ( !this.dragging && !event.pointer.dragging ) {
        this.startDrag( event );
      }
    },

    tryTouchToSnag: function( event ) {
      // allow touches to start a drag by moving "over" this node, and allows clients to specify custom logic for when touchSnag is allowable
      if ( this.options.allowTouchSnag && ( this.options.allowTouchSnag === true || this.options.allowTouchSnag( event ) ) ) {
        this.tryToSnag( event );
      }
    },

    /*---------------------------------------------------------------------------*
     * events called from the node input listener
     *----------------------------------------------------------------------------*/

    // mouse/touch down on this node
    down: function( event ) {
      this.tryToSnag( event );
    },

    // touch enters this node
    touchenter: function( event ) {
      this.tryTouchToSnag( event );
    },

    // touch moves over this node
    touchmove: function( event ) {
      this.tryTouchToSnag( event );
    }
  } );

  return SimpleDragHandler;
} );



// Copyright 2013-2016, University of Colorado Boulder

/**
 * Basic dragging for a node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @deprecated - please use DragListener for new code
 */
define( function( require ) {
  'use strict';

  // modules
  var Emitter = require( 'AXON/Emitter' );
  var EmitterIO = require( 'AXON/EmitterIO' );
  var Vector2IO = require( 'DOT/Vector2IO' );
  var VoidIO = require( 'TANDEM/types/VoidIO' );
  var BooleanProperty = require( 'AXON/BooleanProperty' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Mouse = require( 'SCENERY/input/Mouse' );
  var PhetioObject = require( 'TANDEM/PhetioObject' );
  var scenery = require( 'SCENERY/scenery' );
  var Tandem = require( 'TANDEM/Tandem' );
  var Touch = require( 'SCENERY/input/Touch' );
  var Vector2 = require( 'DOT/Vector2' );

  /**
   * @param {Object} [options]
   * @constructor
   */
  function SimpleDragHandler( options ) {
    var self = this;

    options = _.extend( {

      start: null, // {null|function(Event,Trail)} called when a drag is started
      drag: null, // {null|function(Event,Trail)} called when pointer moves
      end: null,  // {null|function(Event,Trail)} called when a drag is ended

      // {null|function} Called when the pointer moves.
      // Signature is translate( delta: Vector2, oldPosition: Vector2, position: Vector2 )
      translate: null, //

      allowTouchSnag: false,

      // allow changing the mouse button that activates the drag listener.
      // -1 should activate on any mouse button, 0 on left, 1 for middle, 2 for right, etc.
      mouseButton: 0,

      // while dragging with the mouse, sets the cursor to this value
      // (or use null to not override the cursor while dragging)
      dragCursor: 'pointer',

      // when set to true, the handler will get "attached" to a pointer during use, preventing the pointer from starting
      // a drag via something like PressListener
      attach: false,

      // phetio
      tandem: Tandem.required,
      phetioState: false,
      phetioEventType: 'user'

    }, options );
    this.options = options; // @private

    // @public (read-only) {BooleanProperty} - indicates whether dragging is in progress
    this.isDraggingProperty = new BooleanProperty( false, {
      phetioReadOnly: true,
      phetioState: false,
      tandem: options.tandem.createTandem( 'isDraggingProperty' ),
      phetioDocumentation: 'Indicates whether the object is dragging'
    } );

    this.pointer = null;              // the pointer doing the current dragging
    this.trail = null;                // stores the path to the node that is being dragged
    this.transform = null;            // transform of the trail to our node (but not including our node, so we can prepend the deltas)
    this.node = null;                 // the node that we are handling the drag for
    this.lastDragPoint = null;        // the location of the drag at the previous event (so we can calculate a delta)
    this.startTransformMatrix = null; // the node's transform at the start of the drag, so we can reset on a touch cancel
    this.mouseButton = undefined;     // tracks which mouse button was pressed, so we can handle that specifically

    // @public {boolean} - This will be set to true for endDrag calls that are the result of the listener being
    // interrupted. It will be set back to false after the endDrag is finished.
    this.interrupted = false;

    // TODO: consider mouse buttons as separate pointers?

    // @private {Pointer|null} - There are cases like https://github.com/phetsims/equality-explorer/issues/97 where if
    // a touchenter starts a drag that is IMMEDIATELY interrupted, the touchdown would start another drag. We record
    // interruptions here so that we can prevent future enter/down events from the same touch pointer from triggering
    // another startDrag.
    this.lastInterruptedTouchPointer = null;

    // @private {boolean}
    this.disposed = false;

    // @private
    this.dragStartedEmitter = new Emitter( {
      tandem: options.tandem.createTandem( 'dragStartedEmitter' ),
      phetioType: EmitterIO(
        [ { name: 'point', type: Vector2IO, documentation: 'the position of the drag start in view coordinates' },
          { name: 'event', type: VoidIO, documentation: 'the scenery pointer Event' } ] ),
      listener: function( point, event ) {
        if ( self.options.start ) {
          self.options.start.call( null, event, self.trail );
        }
      }
    } );

    // @private
    this.draggedEmitter = new Emitter( {
      phetioHighFrequency: true,
      tandem: options.tandem.createTandem( 'draggedEmitter' ),
      phetioType: EmitterIO(
        [ { name: 'point', type: Vector2IO, documentation: 'the position of the drag in view coordinates' },
          {
            name: 'delta',
            type: Vector2IO,
            documentation: 'the change from the previous position to the current position'
          },
          { name: 'event', type: VoidIO, documentation: 'the scenery pointer Event' } ] ),
      listener: function( point, delta, event ) {

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

          // TODO: add the position in to the listener
          var saveCurrentTarget = event.currentTarget;
          event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          self.options.drag.call( null, event, self.trail ); // new position (old position?) delta
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }
      }
    } );

    // @private
    this.dragEndedEmitter = new Emitter( {
      tandem: options.tandem.createTandem( 'dragEndedEmitter' ),
      phetioType: EmitterIO(
        [ { name: 'point', type: Vector2IO, documentation: 'the position of the drag end in view coordinates' },
          { name: 'event', type: VoidIO, documentation: 'the scenery pointer Event' } ] ),
      listener: function( point, event ) {
        if ( self.options.end ) {

          // drag end may be triggered programmatically and hence event and trail may be undefined
          self.options.end.call( null, event, self.trail );
        }
      }
    } );

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
        if ( !self.dragging || self.disposed ) { return; }

        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler (pointer) up for ' + self.trail.toString() );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === self.pointer, 'Wrong pointer in up' );
        if ( !( event.pointer instanceof Mouse ) || event.domEvent.button === self.mouseButton ) {
          var saveCurrentTarget = event.currentTarget;
          event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          self.endDrag( event );
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // touch cancel
      cancel: function( event ) {
        if ( !self.dragging || self.disposed ) { return; }

        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler (pointer) cancel for ' + self.trail.toString() );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === self.pointer, 'Wrong pointer in cancel' );

        var saveCurrentTarget = event.currentTarget;
        event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
        self.endDrag( event );
        event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget

        // since it's a cancel event, go back!
        if ( !self.transform ) {
          self.node.setMatrix( self.startTransformMatrix );
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // mouse/touch move
      move: function( event ) {
        if ( !self.dragging || self.disposed ) { return; }

        assert && assert( event.pointer === self.pointer, 'Wrong pointer in move' );

        var globalDelta = self.pointer.point.minus( self.lastDragPoint );

        // ignore move events that have 0-length (Chrome seems to be auto-firing these on Windows, see https://code.google.com/p/chromium/issues/detail?id=327114)
        if ( globalDelta.magnitudeSquared() === 0 ) {
          return;
        }

        var delta = self.transform.inverseDelta2( globalDelta );

        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler (pointer) move for ' + self.trail.toString() );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        self.draggedEmitter.emit( event.pointer.point, delta, event );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // pointer interruption
      interrupt: () => {
        self.interrupt();
      }
    };
    PhetioObject.call( this, options );
  }

  scenery.register( 'SimpleDragHandler', SimpleDragHandler );

  return inherit( PhetioObject, SimpleDragHandler, {

    // @private
    get dragging() {
      return this.isDraggingProperty.get();
    },

    set dragging( d ) {
      assert && assert( 'illegal call to set dragging on SimpleDragHandler' );
    },
    startDrag: function( event ) {
      if ( this.dragging ) { return; }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler startDrag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // set a flag on the pointer so it won't pick up other nodes
      event.pointer.dragging = true;
      event.pointer.cursor = this.options.dragCursor;
      event.pointer.addInputListener( this.dragListener, this.options.attach );

      // set all of our persistent information
      this.isDraggingProperty.set( true );
      this.pointer = event.pointer;
      this.trail = event.trail.subtrailTo( event.currentTarget, true );
      this.transform = this.trail.getTransform();
      this.node = event.currentTarget;
      this.lastDragPoint = event.pointer.point;
      this.startTransformMatrix = event.currentTarget.getMatrix().copy();
      // event.domEvent may not exist if this is touch-to-snag
      this.mouseButton = event.pointer instanceof Mouse ? event.domEvent.button : undefined;

      this.dragStartedEmitter.emit( event.pointer.point, event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    endDrag: function( event ) {
      if ( !this.dragging ) { return; }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler endDrag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.pointer.dragging = false;
      this.pointer.cursor = null;
      this.pointer.removeInputListener( this.dragListener );

      this.isDraggingProperty.set( false );

      // Signify drag ended.  In the case of programmatically ended drags, signify drag ended at 0,0.
      // see https://github.com/phetsims/ph-scale-basics/issues/43
      this.dragEndedEmitter.emit( event ? event.pointer.point : Vector2.ZERO, event );

      // release our reference
      this.pointer = null;

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    // Called when input is interrupted on this listener, see https://github.com/phetsims/scenery/issues/218
    interrupt: function() {
      if ( this.dragging ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler interrupt' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.interrupted = true;

        if ( this.pointer instanceof Touch ) {
          this.lastInterruptedTouchPointer = this.pointer;
        }

        // We create a synthetic event here, as there is no available event here.
        this.endDrag( {
          pointer: this.pointer,
          currentTarget: this.node
        } );

        this.interrupted = false;

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    },

    tryToSnag: function( event ) {
      // don't allow drag attempts that use the wrong mouse button (-1 indicates any mouse button works)
      if ( event.pointer instanceof Mouse &&
           event.domEvent &&
           this.options.mouseButton !== event.domEvent.button &&
           this.options.mouseButton !== -1 ) {
        return;
      }

      // If we're disposed, we can't start new drags.
      if ( this.disposed ) {
        return;
      }

      // only start dragging if the pointer isn't dragging anything, we aren't being dragged, and if it's a mouse it's button is down
      if ( !this.dragging &&
           !event.pointer.dragging &&
           event.pointer !== this.lastInterruptedTouchPointer &&
           event.canStartPress() ) {
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
    },

    /**
     * Disposes this listener, releasing any references it may have to a pointer.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler dispose' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.disposed = true;

      if ( this.dragging ) {
        this.pointer.dragging = false;
        this.pointer.cursor = null;
        this.pointer.removeInputListener( this.dragListener );
      }
      this.isDraggingProperty.dispose();

      // It seemed without disposing these led to a memory leak in Energy Skate Park: Basics
      this.dragEndedEmitter.dispose();
      this.draggedEmitter.dispose();
      this.dragStartedEmitter.dispose();

      PhetioObject.prototype.dispose.call( this );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  }, {

    /**
     * Creates an input listener that forwards events to the specified input listener
     * See https://github.com/phetsims/scenery/issues/639
     * @param {function(Event)} down - down function to be added to the input listener
     * @param {Object} [options]
     * @returns {Object} a scenery input listener
     */
    createForwardingListener: function( down, options ) {

      options = _.extend( {
        allowTouchSnag: false
      }, options );

      return {
        down: function( event ) {
          if ( !event.pointer.dragging && event.canStartPress() ) {
            down( event );
          }
        },
        touchenter: function( event ) {
          options.allowTouchSnag && this.down( event );
        },
        touchmove: function( event ) {
          options.allowTouchSnag && this.down( event );
        }
      };
    }
  } );
} );



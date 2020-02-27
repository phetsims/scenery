// Copyright 2013-2020, University of Colorado Boulder

/**
 * Basic dragging for a node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @deprecated - please use DragListener for new code
 */

import Action from '../../../axon/js/Action.js';
import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Vector2IO from '../../../dot/js/Vector2IO.js';
import inherit from '../../../phet-core/js/inherit.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import scenery from '../scenery.js';
import Mouse from './Mouse.js';
import Pointer from './Pointer.js';
import SceneryEvent from './SceneryEvent.js';
import Touch from './Touch.js';

/**
 * @param {Object} [options]
 * @constructor
 */
function SimpleDragHandler( options ) {
  const self = this;

  options = merge( {

    start: null, // {null|function(SceneryEvent,Trail)} called when a drag is started
    drag: null, // {null|function(SceneryEvent,Trail)} called when pointer moves
    end: null,  // {null|function(SceneryEvent,Trail)} called when a drag is ended

    // {null|function} Called when the pointer moves.
    // Signature is translate( { delta: Vector2, oldPosition: Vector2, position: Vector2 } )
    translate: null, //

    // {boolean|function:boolean}
    allowTouchSnag: false,

    // allow changing the mouse button that activates the drag listener.
    // -1 should activate on any mouse button, 0 on left, 1 for middle, 2 for right, etc.
    mouseButton: 0,

    // while dragging with the mouse, sets the cursor to this value
    // (or use null to not override the cursor while dragging)
    dragCursor: 'pointer',

    // when set to true, the handler will get "attached" to a pointer during use, preventing the pointer from starting
    // a drag via something like PressListener
    attach: true,

    // phetio
    tandem: Tandem.REQUIRED,
    phetioState: false,
    phetioEventType: EventType.USER,
    phetioReadOnly: true

  }, options );
  this.options = options; // @private

  // @public (read-only) {BooleanProperty} - indicates whether dragging is in progress
  this.isDraggingProperty = new BooleanProperty( false, {
    phetioReadOnly: options.phetioReadOnly,
    phetioState: false,
    tandem: options.tandem.createTandem( 'isDraggingProperty' ),
    phetioDocumentation: 'Indicates whether the object is dragging'
  } );

  // @public {Pointer|null} - the pointer doing the current dragging
  this.pointer = null;

  // @public {Trail|null} - stores the path to the node that is being dragged
  this.trail = null;

  // @public {Transform3|null} - transform of the trail to our node (but not including our node, so we can prepend
  // the deltas)
  this.transform = null;

  // @public {Node|null} - the node that we are handling the drag for
  this.node = null;

  // @protected {Vector2|null} - the location of the drag at the previous event (so we can calculate a delta)
  this.lastDragPoint = null;

  // @protected {Matrix3|null} - the node's transform at the start of the drag, so we can reset on a touch cancel
  this.startTransformMatrix = null;

  // @public {number|undefined} - tracks which mouse button was pressed, so we can handle that specifically
  this.mouseButton = undefined;

  // @public {boolean} - This will be set to true for endDrag calls that are the result of the listener being
  // interrupted. It will be set back to false after the endDrag is finished.
  this.interrupted = false;

  // @private {Pointer|null} - There are cases like https://github.com/phetsims/equality-explorer/issues/97 where if
  // a touchenter starts a drag that is IMMEDIATELY interrupted, the touchdown would start another drag. We record
  // interruptions here so that we can prevent future enter/down events from the same touch pointer from triggering
  // another startDrag.
  this.lastInterruptedTouchPointer = null;

  // @private {Action}
  this.dragStartAction = new Action( function( point, event ) {

    if ( self.dragging ) { return; }

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler startDrag' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // set a flag on the pointer so it won't pick up other nodes
    event.pointer.dragging = true;
    event.pointer.cursor = self.options.dragCursor;
    event.pointer.addInputListener( self.dragListener, self.options.attach );

    // mark the Intent of this pointer listener to indicate that we want to drag and therefore potentially
    // change the behavior of other listeners in the dispatch phase
    event.pointer.setIntent( Pointer.Intent.DRAG );

    // set all of our persistent information
    self.isDraggingProperty.set( true );
    self.pointer = event.pointer;
    self.trail = event.trail.subtrailTo( event.currentTarget, true );
    self.transform = self.trail.getTransform();
    self.node = event.currentTarget;
    self.lastDragPoint = event.pointer.point;
    self.startTransformMatrix = event.currentTarget.getMatrix().copy();
    // event.domEvent may not exist for touch-to-snag
    self.mouseButton = event.pointer instanceof Mouse ? event.domEvent.button : undefined;

    if ( self.options.start ) {
      self.options.start.call( null, event, self.trail );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }, {
    tandem: options.tandem.createTandem( 'dragStartAction' ),
    phetioReadOnly: options.phetioReadOnly,
    parameters: [ {
      name: 'point',
      phetioType: Vector2IO,
      phetioDocumentation: 'the position of the drag start in view coordinates'
    }, {
      phetioPrivate: true,
      valueType: [ SceneryEvent, null ]
    } ]
  } );

  // @private {Action}
  this.dragAction = new Action( function( point, event ) {

    if ( !self.dragging || self.isDisposed ) { return; }

    const globalDelta = self.pointer.point.minus( self.lastDragPoint );

    // ignore move events that have 0-length. Chrome seems to be auto-firing these on Windows,
    // see https://code.google.com/p/chromium/issues/detail?id=327114
    if ( globalDelta.magnitudeSquared === 0 ) {
      return;
    }

    const delta = self.transform.inverseDelta2( globalDelta );

    assert && assert( event.pointer === self.pointer, 'Wrong pointer in move' );

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler (pointer) move for ' + self.trail.toString() );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // move by the delta between the previous point, using the precomputed transform
    // prepend the translation on the node, so we can ignore whatever other transform state the node has
    if ( self.options.translate ) {
      const translation = self.node.getMatrix().getTranslation();
      self.options.translate.call( null, {
        delta: delta,
        oldPosition: translation,
        position: translation.plus( delta )
      } );
    }
    self.lastDragPoint = self.pointer.point;

    if ( self.options.drag ) {

      // TODO: add the position in to the listener
      const saveCurrentTarget = event.currentTarget;
      event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
      self.options.drag.call( null, event, self.trail ); // new position (old position?) delta
      event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }, {
    phetioHighFrequency: true,
    phetioReadOnly: options.phetioReadOnly,
    tandem: options.tandem.createTandem( 'dragAction' ),
    parameters: [ {
      name: 'point',
      phetioType: Vector2IO,
      phetioDocumentation: 'the position of the drag in view coordinates'
    }, {
      phetioPrivate: true,
      valueType: [ SceneryEvent, null ]
    } ]
  } );

  // @private {Action}
  this.dragEndAction = new Action( function( point, event ) {

    if ( !self.dragging ) { return; }

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler endDrag' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    self.pointer.dragging = false;
    self.pointer.cursor = null;
    self.pointer.removeInputListener( self.dragListener );

    self.isDraggingProperty.set( false );

    if ( self.options.end ) {

      // drag end may be triggered programmatically and hence event and trail may be undefined
      self.options.end.call( null, event, self.trail );
    }

    // release our reference
    self.pointer = null;

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }, {
    tandem: options.tandem.createTandem( 'dragEndAction' ),
    phetioReadOnly: options.phetioReadOnly,
    parameters: [ {
      name: 'point',
      phetioType: Vector2IO,
      phetioDocumentation: 'the position of the drag end in view coordinates'
    }, {
      phetioPrivate: true,
      isValidValue: value => {
        return value === null || value instanceof SceneryEvent ||

               // When interrupted, an object literal is used to signify the interruption,
               // see SimpleDragHandler.interrupt
               ( value.pointer && value.currentTarget );
      }
    }
    ]
  } );

  // @protected {function} - if an ancestor is transformed, pin our node
  this.transformListener = {
    transform: function( args ) {
      if ( !self.trail.isExtensionOf( args.trail, true ) ) {
        return;
      }

      const newMatrix = args.trail.getMatrix();
      const oldMatrix = self.transform.getMatrix();

      // if A was the trail's old transform, B is the trail's new transform, we need to apply (B^-1 A) to our node
      self.node.prependMatrix( newMatrix.inverted().timesMatrix( oldMatrix ) );

      // store the new matrix so we can do deltas using it now
      self.transform.setMatrix( newMatrix );
    }
  };

  // @protected {function} - this listener gets added to the pointer when it starts dragging our node
  this.dragListener = {
    // mouse/touch up
    up: function( event ) {
      if ( !self.dragging || self.isDisposed ) { return; }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler (pointer) up for ' + self.trail.toString() );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( event.pointer === self.pointer, 'Wrong pointer in up' );
      if ( !( event.pointer instanceof Mouse ) || event.domEvent.button === self.mouseButton ) {
        const saveCurrentTarget = event.currentTarget;
        event.currentTarget = self.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
        self.endDrag( event );
        event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    // touch cancel
    cancel: function( event ) {
      if ( !self.dragging || self.isDisposed ) { return; }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler (pointer) cancel for ' + self.trail.toString() );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( event.pointer === self.pointer, 'Wrong pointer in cancel' );

      const saveCurrentTarget = event.currentTarget;
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
      self.dragAction.execute( event.pointer.point, event );
    }
  };
  PhetioObject.call( this, options );
}

scenery.register( 'SimpleDragHandler', SimpleDragHandler );

export default inherit( PhetioObject, SimpleDragHandler, {

  // @private
  get dragging() {
    return this.isDraggingProperty.get();
  },

  set dragging( d ) {
    assert && assert( 'illegal call to set dragging on SimpleDragHandler' );
  },
  startDrag: function( event ) {
    this.dragStartAction.execute( event.pointer.point, event );
  },

  endDrag: function( event ) {

    // Signify drag ended.  In the case of programmatically ended drags, signify drag ended at 0,0.
    // see https://github.com/phetsims/ph-scale-basics/issues/43
    this.dragEndAction.execute( event ? event.pointer.point : Vector2.ZERO, event );
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
    if ( this.isDisposed ) {
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

    if ( this.dragging ) {
      this.pointer.dragging = false;
      this.pointer.cursor = null;
      this.pointer.removeInputListener( this.dragListener );
    }
    this.isDraggingProperty.dispose();

    // It seemed without disposing these led to a memory leak in Energy Skate Park: Basics
    this.dragEndAction.dispose();
    this.dragAction.dispose();
    this.dragStartAction.dispose();

    PhetioObject.prototype.dispose.call( this );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }
}, {

  /**
   * Creates an input listener that forwards events to the specified input listener
   * See https://github.com/phetsims/scenery/issues/639
   * @param {function(SceneryEvent)} down - down function to be added to the input listener
   * @param {Object} [options]
   * @returns {Object} a scenery input listener
   */
  createForwardingListener: function( down, options ) {

    options = merge( {
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
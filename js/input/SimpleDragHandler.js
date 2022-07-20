// Copyright 2013-2022, University of Colorado Boulder

/**
 * Basic dragging for a node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Vector2 from '../../../dot/js/Vector2.js';
import deprecationWarning from '../../../phet-core/js/deprecationWarning.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioAction from '../../../tandem/js/PhetioAction.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { Mouse, scenery, SceneryEvent } from '../imports.js';

/**
 * @deprecated - please use DragListener for new code
 */
class SimpleDragHandler extends PhetioObject {
  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    assert && deprecationWarning( 'SimpleDragHandler is deprecated, please use DragListener instead' );

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

    super();

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
    this.lastInterruptedTouchLikePointer = null;

    // @private {boolean}
    this._attach = options.attach;

    // @private {Action}
    this.dragStartAction = new PhetioAction( ( point, event ) => {

      if ( this.dragging ) { return; }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler startDrag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // set a flag on the pointer so it won't pick up other nodes
      if ( this._attach ) {
        // Only set the `dragging` flag on the pointer if we have attach:true
        // See https://github.com/phetsims/scenery/issues/206
        event.pointer.dragging = true;
      }
      event.pointer.cursor = this.options.dragCursor;
      event.pointer.addInputListener( this.dragListener, this.options.attach );

      // mark the Intent of this pointer listener to indicate that we want to drag and therefore potentially
      // change the behavior of other listeners in the dispatch phase
      event.pointer.reserveForDrag();

      // set all of our persistent information
      this.isDraggingProperty.set( true );
      this.pointer = event.pointer;
      this.trail = event.trail.subtrailTo( event.currentTarget, true );
      this.transform = this.trail.getTransform();
      this.node = event.currentTarget;
      this.lastDragPoint = event.pointer.point;
      this.startTransformMatrix = event.currentTarget.getMatrix().copy();
      // event.domEvent may not exist for touch-to-snag
      this.mouseButton = event.pointer instanceof Mouse ? event.domEvent.button : undefined;

      if ( this.options.start ) {
        this.options.start.call( null, event, this.trail );
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }, {
      tandem: options.tandem.createTandem( 'dragStartAction' ),
      phetioReadOnly: options.phetioReadOnly,
      parameters: [ {
        name: 'point',
        phetioType: Vector2.Vector2IO,
        phetioDocumentation: 'the position of the drag start in view coordinates'
      }, {
        phetioPrivate: true,
        valueType: [ SceneryEvent, null ]
      } ]
    } );

    // @private {Action}
    this.dragAction = new PhetioAction( ( point, event ) => {

      if ( !this.dragging || this.isDisposed ) { return; }

      const globalDelta = this.pointer.point.minus( this.lastDragPoint );

      // ignore move events that have 0-length. Chrome seems to be auto-firing these on Windows,
      // see https://code.google.com/p/chromium/issues/detail?id=327114
      if ( globalDelta.magnitudeSquared === 0 ) {
        return;
      }

      const delta = this.transform.inverseDelta2( globalDelta );

      assert && assert( event.pointer === this.pointer, 'Wrong pointer in move' );

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `SimpleDragHandler (pointer) move for ${this.trail.toString()}` );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // move by the delta between the previous point, using the precomputed transform
      // prepend the translation on the node, so we can ignore whatever other transform state the node has
      if ( this.options.translate ) {
        const translation = this.node.getMatrix().getTranslation();
        this.options.translate.call( null, {
          delta: delta,
          oldPosition: translation,
          position: translation.plus( delta )
        } );
      }
      this.lastDragPoint = this.pointer.point;

      if ( this.options.drag ) {

        // TODO: add the position in to the listener
        const saveCurrentTarget = event.currentTarget;
        event.currentTarget = this.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
        this.options.drag.call( null, event, this.trail ); // new position (old position?) delta
        event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }, {
      phetioHighFrequency: true,
      phetioReadOnly: options.phetioReadOnly,
      tandem: options.tandem.createTandem( 'dragAction' ),
      parameters: [ {
        name: 'point',
        phetioType: Vector2.Vector2IO,
        phetioDocumentation: 'the position of the drag in view coordinates'
      }, {
        phetioPrivate: true,
        valueType: [ SceneryEvent, null ]
      } ]
    } );

    // @private {Action}
    this.dragEndAction = new PhetioAction( ( point, event ) => {

      if ( !this.dragging ) { return; }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler endDrag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      if ( this._attach ) {
        // Only set the `dragging` flag on the pointer if we have attach:true
        // See https://github.com/phetsims/scenery/issues/206
        this.pointer.dragging = false;
      }
      this.pointer.cursor = null;
      this.pointer.removeInputListener( this.dragListener );

      this.isDraggingProperty.set( false );

      if ( this.options.end ) {

        // drag end may be triggered programmatically and hence event and trail may be undefined
        this.options.end.call( null, event, this.trail );
      }

      // release our reference
      this.pointer = null;

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }, {
      tandem: options.tandem.createTandem( 'dragEndAction' ),
      phetioReadOnly: options.phetioReadOnly,
      parameters: [ {
        name: 'point',
        phetioType: Vector2.Vector2IO,
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
      transform: args => {
        if ( !this.trail.isExtensionOf( args.trail, true ) ) {
          return;
        }

        const newMatrix = args.trail.getMatrix();
        const oldMatrix = this.transform.getMatrix();

        // if A was the trail's old transform, B is the trail's new transform, we need to apply (B^-1 A) to our node
        this.node.prependMatrix( newMatrix.inverted().timesMatrix( oldMatrix ) );

        // store the new matrix so we can do deltas using it now
        this.transform.setMatrix( newMatrix );
      }
    };

    // @protected {function} - this listener gets added to the pointer when it starts dragging our node
    this.dragListener = {
      // mouse/touch up
      up: event => {
        if ( !this.dragging || this.isDisposed ) { return; }

        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `SimpleDragHandler (pointer) up for ${this.trail.toString()}` );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === this.pointer, 'Wrong pointer in up' );
        if ( !( event.pointer instanceof Mouse ) || event.domEvent.button === this.mouseButton ) {
          const saveCurrentTarget = event.currentTarget;
          event.currentTarget = this.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
          this.endDrag( event );
          event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // touch cancel
      cancel: event => {
        if ( !this.dragging || this.isDisposed ) { return; }

        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `SimpleDragHandler (pointer) cancel for ${this.trail.toString()}` );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === this.pointer, 'Wrong pointer in cancel' );

        const saveCurrentTarget = event.currentTarget;
        event.currentTarget = this.node; // #66: currentTarget on a pointer is null, so set it to the node we're dragging
        this.endDrag( event );
        event.currentTarget = saveCurrentTarget; // be polite to other listeners, restore currentTarget

        // since it's a cancel event, go back!
        if ( !this.transform ) {
          this.node.setMatrix( this.startTransformMatrix );
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // mouse/touch move
      move: event => {
        this.dragAction.execute( event.pointer.point, event );
      },

      // pointer interruption
      interrupt: () => {
        this.interrupt();
      }
    };

    this.initializePhetioObject( {}, options );
  }

  // @private
  get dragging() {
    return this.isDraggingProperty.get();
  }

  set dragging( d ) {
    assert && assert( false, 'illegal call to set dragging on SimpleDragHandler' );
  }

  /**
   * @public
   *
   * @param {SceneryEvent} event
   */
  startDrag( event ) {
    this.dragStartAction.execute( event.pointer.point, event );
  }

  /**
   * @public
   *
   * @param {SceneryEvent} event
   */
  endDrag( event ) {

    // Signify drag ended.  In the case of programmatically ended drags, signify drag ended at 0,0.
    // see https://github.com/phetsims/ph-scale-basics/issues/43
    this.dragEndAction.execute( event ? event.pointer.point : Vector2.ZERO, event );
  }

  /**
   * Called when input is interrupted on this listener, see https://github.com/phetsims/scenery/issues/218
   * @public
   */
  interrupt() {
    if ( this.dragging ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler interrupt' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.interrupted = true;

      if ( this.pointer && this.pointer.isTouchLike() ) {
        this.lastInterruptedTouchLikePointer = this.pointer;
      }

      // We create a synthetic event here, as there is no available event here.
      this.endDrag( {
        pointer: this.pointer,
        currentTarget: this.node
      } );

      this.interrupted = false;

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  }

  /**
   * @public
   *
   * @param {SceneryEvent} event
   */
  tryToSnag( event ) {
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
         // Don't check pointer.dragging if we don't attach, see https://github.com/phetsims/scenery/issues/206
         ( !event.pointer.dragging || !this._attach ) &&
         event.pointer !== this.lastInterruptedTouchLikePointer &&
         event.canStartPress() ) {
      this.startDrag( event );
    }
  }

  /**
   * @public
   *
   * @param {SceneryEvent} event
   */
  tryTouchToSnag( event ) {
    // allow touches to start a drag by moving "over" this node, and allows clients to specify custom logic for when touchSnag is allowable
    if ( this.options.allowTouchSnag && ( this.options.allowTouchSnag === true || this.options.allowTouchSnag( event ) ) ) {
      this.tryToSnag( event );
    }
  }

  /*---------------------------------------------------------------------------*
   * events called from the node input listener
   *----------------------------------------------------------------------------*/

  /**
   * Event listener method - mouse/touch down on this node
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  down( event ) {
    this.tryToSnag( event );
  }

  /**
   * Event listener method - touch enters this node
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  touchenter( event ) {
    this.tryTouchToSnag( event );
  }

  /**
   * Event listener method - touch moves over this node
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  touchmove( event ) {
    this.tryTouchToSnag( event );
  }

  /**
   * Disposes this listener, releasing any references it may have to a pointer.
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'SimpleDragHandler dispose' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    if ( this.dragging ) {
      if ( this._attach ) {
        // Only set the `dragging` flag on the pointer if we have attach:true
        // See https://github.com/phetsims/scenery/issues/206
        this.pointer.dragging = false;
      }
      this.pointer.cursor = null;
      this.pointer.removeInputListener( this.dragListener );
    }
    this.isDraggingProperty.dispose();

    // It seemed without disposing these led to a memory leak in Energy Skate Park: Basics
    this.dragEndAction.dispose();
    this.dragAction.dispose();
    this.dragStartAction.dispose();

    super.dispose();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }


  /**
   * Creates an input listener that forwards events to the specified input listener
   * @public
   *
   * See https://github.com/phetsims/scenery/issues/639
   *
   * @param {function(SceneryEvent)} down - down function to be added to the input listener
   * @param {Object} [options]
   * @returns {Object} a scenery input listener
   */
  static createForwardingListener( down, options ) {

    options = merge( {
      allowTouchSnag: false
    }, options );

    return {
      down: event => {
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
}

scenery.register( 'SimpleDragHandler', SimpleDragHandler );

export default SimpleDragHandler;
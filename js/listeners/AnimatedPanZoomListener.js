// Copyright 2019-2023, University of Colorado Boulder

/**
 * A PanZoomListener that supports additional forms of input for pan and zoom, including trackpad gestures, mouse
 * wheel, and keyboard input. These gestures will animate the target node to its destination translation and scale so it
 * uses a step function that must be called every animation frame.
 *
 * @author Jesse Greenberg
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Utils from '../../../dot/js/Utils.js';
import Vector2 from '../../../dot/js/Vector2.js';
import merge from '../../../phet-core/js/merge.js';
import platform from '../../../phet-core/js/platform.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioAction from '../../../tandem/js/PhetioAction.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { EventIO, FocusManager, globalKeyStateTracker, Intent, KeyboardDragListener, KeyboardUtils, KeyboardZoomUtils, Node, PanZoomListener, PDOMPointer, PDOMUtils, PressListener, scenery } from '../imports.js';

// constants
const MOVE_CURSOR = 'all-scroll';
const MAX_SCROLL_VELOCITY = 150; // max global view coords per second while scrolling with middle mouse button drag

// The max speed of translation when animating from source position to destination position in the coordinate frame
// of the parent of the targetNode of this listener. Increase the value of this to animate faster to the destination
// position when panning to targets.
const MAX_TRANSLATION_SPEED = 1000;

// scratch variables to reduce garbage
const scratchTranslationVector = new Vector2( 0, 0 );
const scratchScaleTargetVector = new Vector2( 0, 0 );
const scratchVelocityVector = new Vector2( 0, 0 );
const scratchBounds = new Bounds2( 0, 0, 0, 0 );

class AnimatedPanZoomListener extends PanZoomListener {

  /**
   * @param {Node} targetNode - Node to be transformed by this listener
   * @param {Object} [options]
   */
  constructor( targetNode, options ) {

    // While this is still JavaScript, this relatively useless assertion lets us import Node for other TypeScript
    assert && assert( targetNode instanceof Node );

    options = merge( {
      tandem: Tandem.REQUIRED
    }, options );

    super( targetNode, options );

    // @private {null|Vector2} - This point is the center of the transformedPanBounds (see PanZoomListener) in
    // the parent coordinate frame of the targetNode. This is the current center of the transformedPanBounds, and
    // during animation we will move this point closer to the destinationPosition.
    this.sourcePosition = null;

    // @private {null|Vector2} - The destination for translation, we will reposition the targetNode until the
    // sourcePosition matches this point. This is in the parent coordinate frame of the targetNode.
    this.destinationPosition = null;

    // @private {number} - The current scale of the targetNode. During animation we will scale the targetNode until
    // this matches the destinationScale.
    this.sourceScale = this.getCurrentScale();

    // @private {number} - The desired scale for the targetNode, the node is repositioned until sourceScale matches
    // destinationScale.
    this.destinationScale = this.getCurrentScale();

    // @private {null|Vector2} - The point at which a scale gesture was initiated. This is usually the mouse point in
    // the global coordinate frame when a wheel or trackpad zoom gesture is initiated. The targetNode will appear to
    // be zoomed into this point. This is in the global coordinate frame.
    this.scaleGestureTargetPosition = null;

    // @private {Array.<number>} - scale changes in discrete amounts for certain types of input, and in these
    // cases this array defines the discrete scales possible
    this.discreteScales = calculateDiscreteScales( this._minScale, this._maxScale );

    // @private {MiddlePress|null} - If defined, indicates that a middle mouse button is down to pan in the direction
    // of cursor movement.
    this.middlePress = null;

    // @private {Bounds2|null} - these bounds define behavior of panning during interaction with another listener
    // that declares its intent for dragging. If the pointer is out of these bounds and its intent is for dragging,
    // we will try to reposition so that the dragged object remains visible
    this._dragBounds = null;

    // @private {Bounds2} - The panBounds in the local coordinate frame of the targetNode. Generally, these are the
    // bounds of the targetNode that you can see within the panBounds.
    this._transformedPanBounds = this._panBounds.transformed( this._targetNode.matrix.inverted() );

    // @private - whether or not the Pointer went down within the drag bounds - if it went down out of drag bounds
    // then user likely trying to pull an object back into view so we prevent panning during drag
    this._draggingInDragBounds = false;

    // @private {TInputListener[]} - A collection of listeners Pointers with attached listeners that are down. Used
    // primarily to determine if the attached listener defines any unique behavior that should happen during a drag,
    // such as panning to keep custom Bounds in view. See TInputListener.createPanTargetBounds.
    this._attachedPointers = [];

    // @private {boolean} - Certain calculations can only be done once available pan bounds are finite.
    this.boundsFinite = false;

    // listeners that will be bound to `this` if we are on a (non-touchscreen) safari platform, referenced for
    // removal on dispose
    let boundGestureStartListener = null;
    let boundGestureChangeListener = null;

    // @private {Action} - Action wrapping work to be done when a gesture starts on a macOS trackpad (specific
    // to that platform!). Wrapped in an action so that state is captured for PhET-iO
    this.gestureStartAction = new PhetioAction( domEvent => {
      assert && assert( domEvent instanceof window.Event );
      assert && assert( domEvent.pageX, 'pageX required on DOMEvent' );
      assert && assert( domEvent.pageY, 'pageY required on DOMEvent' );
      assert && assert( domEvent.scale, 'scale required on DOMEvent' );

      // prevent Safari from doing anything native with this gesture
      domEvent.preventDefault();

      this.trackpadGestureStartScale = domEvent.scale;
      this.scaleGestureTargetPosition = new Vector2( domEvent.pageX, domEvent.pageY );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'gestureStartAction' ),
      parameters: [ { name: 'event', phetioType: EventIO } ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Action that executes whenever a gesture starts on a trackpad in macOS Safari.'
    } );

    // @private {Action} - Action wrapping work to be done when gesture changes on a macOS trackpad (specfic to that
    // platform!). Wrapped in an action so state is captured for PhET-iO
    this.gestureChangeAction = new PhetioAction( domEvent => {
      assert && assert( domEvent instanceof window.Event );
      assert && assert( domEvent.scale, 'scale required on DOMEvent' );

      // prevent Safari from changing position or scale natively
      domEvent.preventDefault();

      const newScale = this.sourceScale + domEvent.scale - this.trackpadGestureStartScale;
      this.setDestinationScale( newScale );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'gestureChangeAction' ),
      parameters: [ { name: 'event', phetioType: EventIO } ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Action that executes whenever a gesture changes on a trackpad in macOS Safari.'
    } );

    // respond to macOS trackpad input, but don't respond to this input on an iOS touch screen
    if ( platform.safari && !platform.mobileSafari ) {
      boundGestureStartListener = this.handleGestureStartEvent.bind( this );
      boundGestureChangeListener = this.handleGestureChangeEvent.bind( this );

      // @private {number} - the scale of the targetNode at the start of the gesture, used to calculate
      // how scale to apply from 'gesturechange' event
      this.trackpadGestureStartScale = this.getCurrentScale();

      // WARNING: These events are non-standard, but this is the only way to detect and prevent native trackpad
      // input on macOS Safari. For Apple documentation about these events, see
      // https://developer.apple.com/documentation/webkitjs/gestureevent
      window.addEventListener( 'gesturestart', boundGestureStartListener );
      window.addEventListener( 'gesturechange', boundGestureChangeListener );
    }

    // Handle key input from events outside of the PDOM - in this case it is impossible for the PDOMPointer
    // to be attached so we have free reign over the keyboard
    globalKeyStateTracker.keydownEmitter.addListener( this.windowKeydown.bind( this ) );

    const displayFocusListener = focus => {
      if ( focus && this.getCurrentScale() > 1 ) {
        this.keepTrailInView( focus.trail );
      }
    };
    FocusManager.pdomFocusProperty.link( displayFocusListener );

    // set source and destination positions and scales after setting from state
    // to initialize values for animation with AnimatedPanZoomListener
    this.sourceFramePanBoundsProperty.lazyLink( () => {
      const simGlobal = _.get( window, 'phet.joist.sim', null ); // returns null if global isn't found

      if ( ( simGlobal && simGlobal.isSettingPhetioStateProperty.value ) ) {
        this.initializePositions();
        this.sourceScale = this.getCurrentScale();
        this.setDestinationScale( this.sourceScale );
      }
    }, {

      // guarantee that the matrixProperty value is up to date when this listener is called
      phetioDependencies: [ this.matrixProperty ]
    } );

    // @private - called by dispose
    this.disposeAnimatedPanZoomListener = () => {
      boundGestureStartListener && window.removeEventListener( 'gesturestart', boundGestureStartListener );
      boundGestureChangeListener && window.removeEventListener( 'gestureChange', boundGestureChangeListener );

      FocusManager.pdomFocusProperty.unlink( displayFocusListener );
    };
  }

  /**
   * Step the listener, supporting any animation as the target node is transformed to target position and scale.
   * @public
   *
   * @param {number} dt
   */
  step( dt ) {
    if ( this.middlePress ) {
      this.handleMiddlePress( dt );
    }

    // if dragging an item with a mouse or touch pointer, make sure that it ramains visible in the zoomed in view,
    // panning to it when it approaches edge of the screen
    if ( this._attachedPointers.length > 0 ) {

      // only need to do this work if we are zoomed in
      if ( this.getCurrentScale() > 1 ) {
        if ( this._attachedPointers.length > 0 ) {

          // Filter out any pointers that no longer have an attached listener due to interruption from things like opening
          // the context menu with a right click.
          this._attachedPointers = this._attachedPointers.filter( pointer => pointer.attachedListener );
          assert && assert( this._attachedPointers.length <= 10, 'Not clearing attachedPointers, there is probably a memory leak' );
        }

        // Only reposition if one of the attached pointers is down and dragging within the drag bounds area, or if one
        // of the attached pointers is a PDOMPointer, which indicates that we are dragging with alternative input
        // (in which case draggingInDragBounds does not apply)
        if ( this._draggingInDragBounds || this._attachedPointers.some( pointer => pointer instanceof PDOMPointer ) ) {
          this.repositionDuringDrag();
        }
      }
    }

    this.animateToTargets( dt );
  }

  /**
   * Attach a MiddlePress for drag panning, if detected.
   * @public
   * @override
   *
   * @param {SceneryEvent} event
   */
  down( event ) {
    PanZoomListener.prototype.down.call( this, event );

    // If the Pointer signifies the input is intended for dragging save a reference to the trail so we can support
    // keeping the event target in view during the drag operation.
    if ( this._dragBounds !== null && event.pointer.hasIntent( Intent.DRAG ) ) {

      // if this is our only down pointer, see if we should start panning during drag
      if ( this._attachedPointers.length === 0 ) {
        this._draggingInDragBounds = this._dragBounds.containsPoint( event.pointer.point );
      }

      // All conditions are met to start watching for bounds to keep in view during a drag interaction. Eagerly
      // save the attachedListener here so that we don't have to do any work in the move event.
      if ( event.pointer.attachedListener ) {
        if ( !this._attachedPointers.includes( event.pointer ) ) {
          this._attachedPointers.push( event.pointer );
        }
      }
    }

    // begin middle press panning if we aren't already in that state
    if ( event.pointer.type === 'mouse' && event.pointer.middleDown && !this.middlePress ) {
      this.middlePress = new MiddlePress( event.pointer, event.trail );
      event.pointer.cursor = MOVE_CURSOR;
    }
    else {
      this.cancelMiddlePress();
    }
  }

  /**
   * If in a state where we are panning from a middle mouse press, exit that state.
   * @private
   */
  cancelMiddlePress() {
    if ( this.middlePress ) {
      this.middlePress.pointer.cursor = null;
      this.middlePress = null;

      this.stopInProgressAnimation();
    }
  }

  /**
   * Listener for the attached pointer on move. Only move if a middle press is not currently down.
   * @protected
   * @override
   *
   * @param {SceneryEvent} event
   */
  movePress( event ) {
    if ( !this.middlePress ) {
      PanZoomListener.prototype.movePress.call( this, event );
    }
  }

  /**
   * Part of the Scenery listener API. Supports repositioning while dragging a more descendant level
   * Node under this listener. If the node and pointer are out of the dragBounds, we reposition to keep the Node
   * visible within dragBounds.
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  move( event ) {

    // No need to do this work if we are zoomed out.
    if ( this._attachedPointers.length > 0 && this.getCurrentScale() > 1 ) {

      // Only try to get the attached listener if we didn't successfully get it on the down event. This should only
      // happen if the drag did not start withing dragBounds (the listener is likely pulling the Node into view) or
      // if a listener has not been attached yet. Once a listener is attached we can start using it to look for the
      // bounds to keep in view.
      if ( this._draggingInDragBounds ) {
        if ( !this._attachedPointers.includes( event.pointer ) ) {
          const hasDragIntent = this.hasDragIntent( event.pointer );
          const currentTargetExists = event.currentTarget !== null;

          if ( currentTargetExists && hasDragIntent ) {
            if ( event.pointer.attachedListener ) {
              this._attachedPointers.push( event.pointer );
            }
          }
        }
      }
      else {
        this._draggingInDragBounds = this._dragBounds.containsPoint( event.pointer.point );
      }
    }
  }

  /**
   * Gets the Bounds2 in the global coordinate frame that we are going to try to keep in view during a drag
   * operation.
   * @private
   *
   * @returns {Bounds2|null}
   */
  getGlobalBoundsToViewDuringDrag() {
    let globalBoundsToView = null;

    if ( this._attachedPointers.length > 0 ) {

      // We have an attachedListener from a SceneryEvent Pointer, see if it has information we can use to
      // get the target Bounds for the drag event.

      // Only use the first one so that unique dragging behaviors don't "fight" if multiple pointers are down.
      const activeListener = this._attachedPointers[ 0 ].attachedListener;

      if ( activeListener.createPanTargetBounds ) {

        // client has defined the Bounds they want to keep in view for this Pointer (it is assigned to the
        // Pointer to support multitouch cases)
        globalBoundsToView = activeListener.createPanTargetBounds();
      }
      else if ( activeListener.listener instanceof PressListener ||
                activeListener.listener instanceof KeyboardDragListener ) {
        const attachedPressListener = activeListener.listener;

        // The PressListener might not be pressed anymore but the Pointer is still down, in which case it
        // has been interrupted or cancelled.
        // NOTE: It is possible I need to cancelPanDuringDrag() if it is no longer pressed, but I don't
        // want to clear the reference to the attachedListener, and I want to support resuming drag during touch-snag.
        if ( attachedPressListener.isPressed ) {

          // this will either be the PressListener's targetNode or the default target of the SceneryEvent on press
          const target = attachedPressListener.getCurrentTarget();

          // TODO: For now we cannot support DAG. We may be able to use PressListener.pressedTrail instead of
          // getCurrentTarget, and then we would have a uniquely defined trail. See
          // https://github.com/phetsims/scenery/issues/1361 and
          // https://github.com/phetsims/scenery/issues/1356#issuecomment-1039678678
          if ( target.instances.length === 1 ) {
            globalBoundsToView = target.instances[ 0 ].trail.parentToGlobalBounds( target.visibleBounds );
          }
        }
      }
    }

    return globalBoundsToView;
  }

  /**
   * During a drag of another Node that is a descendant of this listener's targetNode, reposition if the
   * node is out of dragBounds so that the Node is always within panBounds.
   * @private
   */
  repositionDuringDrag() {
    const globalBounds = this.getGlobalBoundsToViewDuringDrag();
    globalBounds && this.keepBoundsInView( globalBounds, this._attachedPointers.some( pointer => pointer instanceof PDOMPointer ) );
  }

  /**
   * Stop panning during drag by clearing variables that are set to indicate and provide information for this work.
   * @param {SceneryEvent} [event] - if not provided all are panning is cancelled and we assume interruption
   * @private
   */
  cancelPanningDuringDrag( event ) {

    if ( event ) {

      // remove the attachedPointer associated with the event
      const index = this._attachedPointers.indexOf( event.pointer );
      if ( index > -1 ) {
        this._attachedPointers.splice( index, 1 );
      }
    }
    else {

      // There is no SceneryEvent, we must be interrupting - clear all attachedPointers
      this._attachedPointers = [];
    }

    // Clear flag indicating we are "dragging in bounds" next move
    this._draggingInDragBounds = false;
  }

  /**
   * Scenery listener API. Cancel any drag and pan behavior for the Pointer on the event.
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  up( event ) {
    this.cancelPanningDuringDrag( event );
  }

  /**
   * Input listener for the 'wheel' event, part of the Scenery Input API.
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  wheel( event ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener wheel' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // cannot reposition if a dragging with middle mouse button - but wheel zoom should not cancel a middle press
    // (behavior copied from other browsers)
    if ( !this.middlePress ) {
      const wheel = new Wheel( event, this._targetScale );
      this.repositionFromWheel( wheel, event );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Keydown listener for events outside of the PDOM. Attached as a listener to the body and driven by
   * Events rather than SceneryEvents. When we handle Events from within the PDOM we need the Pointer to
   * determine if attached. But from outside of the PDOM we know that there is no focus in the document and therfore
   * the PDOMPointer is not attached.
   * @private
   *
   * @param {Event} domEvent
   */
  windowKeydown( domEvent ) {

    // on any keyboard reposition interrupt the middle press panning
    this.cancelMiddlePress();

    const simGlobal = _.get( window, 'phet.joist.sim', null ); // returns null if global isn't found

    if ( !simGlobal || !simGlobal.display._accessible ||
         !simGlobal.display.pdomRootElement.contains( domEvent.target ) ) {
      this.handleZoomCommands( domEvent );

      // handle translation without worry of the pointer being attached because there is no pointer at this level
      if ( KeyboardUtils.isArrowKey( domEvent ) ) {
        const keyPress = new KeyPress( globalKeyStateTracker, this.getCurrentScale(), this._targetScale );
        this.repositionFromKeys( keyPress );
      }
    }
  }

  /**
   * For the Scenery listener API, handle a keydown event. This SceneryEvent will have been dispatched from
   * Input.dispatchEvent and so the Event target must be within the PDOM. In this case, we may
   * need to prevent translation if the PDOMPointer is attached.
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  keydown( event ) {
    const domEvent = event.domEvent;

    // on any keyboard reposition interrupt the middle press panning
    this.cancelMiddlePress();

    // handle zoom
    this.handleZoomCommands( domEvent );

    const keyboardDragIntent = event.pointer.hasIntent( Intent.KEYBOARD_DRAG );

    // handle translation
    if ( KeyboardUtils.isArrowKey( domEvent ) ) {

      if ( !keyboardDragIntent ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener handle arrow key down' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        const keyPress = new KeyPress( globalKeyStateTracker, this.getCurrentScale(), this._targetScale );
        this.repositionFromKeys( keyPress );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    }

    if ( KeyboardUtils.isMovementKey( domEvent ) ) {
      if ( keyboardDragIntent ) {

        // Look for any attached pointers if we are dragging with a keyboard and add them to the list. When dragging
        // stops the Pointer listener is detached and the pointer is removed from the list in `step()`.
        if ( event.pointer.isAttached() ) {
          if ( !this._attachedPointers.includes( event.pointer ) ) {
            this._attachedPointers.push( event.pointer );
          }
        }
      }
    }
  }

  /**
   * Handle zoom commands from a keyboard.
   * @private
   *
   * @param {Event} domEvent
   */
  handleZoomCommands( domEvent ) {

    // handle zoom - Safari doesn't receive the keyup event when the meta key is pressed so we cannot use
    // the keyStateTracker to determine if zoom keys are down
    const zoomInCommandDown = KeyboardZoomUtils.isZoomCommand( domEvent, true );
    const zoomOutCommandDown = KeyboardZoomUtils.isZoomCommand( domEvent, false );

    if ( zoomInCommandDown || zoomOutCommandDown ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiPanZoomListener keyboard zoom in' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // don't allow native browser zoom
      domEvent.preventDefault();

      const nextScale = this.getNextDiscreteScale( zoomInCommandDown );
      const keyPress = new KeyPress( globalKeyStateTracker, nextScale, this._targetScale );
      this.repositionFromKeys( keyPress );
    }
    else if ( KeyboardZoomUtils.isZoomResetCommand( domEvent ) ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener keyboard reset' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // this is a native command, but we are taking over
      domEvent.preventDefault();
      this.resetTransform();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  }

  /**
   * This is just for macOS Safari. Responds to trackpad input. Prevents default browser behavior and sets values
   * required for for repositioning as user operates the track pad.
   * @private
   *
   * @param {Event} domEvent
   */
  handleGestureStartEvent( domEvent ) {
    this.gestureStartAction.execute( domEvent );
  }

  /**
   * This is just for macOS Safari. Responds to trackpad input. Prevends default browser behavior and
   * sets destination scale as user pinches on the trackpad.
   * @private
   *
   * @param {Event} domEvent
   */
  handleGestureChangeEvent( domEvent ) {
    this.gestureChangeAction.execute( domEvent );
  }

  /**
   * Handle the down MiddlePress during animation. If we have a middle press we need to update position target.
   * @private
   *
   * @param {number} dt
   */
  handleMiddlePress( dt ) {
    assert && assert( this.middlePress, 'MiddlePress must be defined to handle' );

    if ( dt > 0 ) {
      const currentPoint = this.middlePress.pointer.point;
      const globalDelta = currentPoint.minus( this.middlePress.initialPoint );

      // magnitude alone is too fast, reduce by a bit
      const reducedMagnitude = globalDelta.magnitude / 100;
      if ( reducedMagnitude > 0 ) {

        // set the delta vector in global coordinates, limited by a maximum view coords/second velocity, corrected
        // for any representative target scale
        globalDelta.setMagnitude( Math.min( reducedMagnitude / dt, MAX_SCROLL_VELOCITY * this._targetScale ) );
        this.setDestinationPosition( this.sourcePosition.plus( globalDelta ) );
      }
    }
  }

  /**
   * Translate and scale to a target point. The result of this function should make it appear that we are scaling
   * in or out of a particular point on the target node. This actually modifies the matrix of the target node. To
   * accomplish zooming into a particular point, we compute a matrix that would transform the target node from
   * the target point, then apply scale, then translate the target back to the target point.
   * @public
   *
   * @param {Vector2} globalPoint - point to zoom in on, in the global coordinate frame
   * @param {number} scaleDelta
   */
  translateScaleToTarget( globalPoint, scaleDelta ) {
    const pointInLocalFrame = this._targetNode.globalToLocalPoint( globalPoint );
    const pointInParentFrame = this._targetNode.globalToParentPoint( globalPoint );

    const fromLocalPoint = Matrix3.translation( -pointInLocalFrame.x, -pointInLocalFrame.y );
    const toTargetPoint = Matrix3.translation( pointInParentFrame.x, pointInParentFrame.y );

    const nextScale = this.limitScale( this.getCurrentScale() + scaleDelta );

    // we first translate from target point, then apply scale, then translate back to target point ()
    // so that it appears as though we are zooming into that point
    const scaleMatrix = toTargetPoint.timesMatrix( Matrix3.scaling( nextScale ) ).timesMatrix( fromLocalPoint );
    this.matrixProperty.set( scaleMatrix );

    // make sure that we are still within PanZoomListener constraints
    this.correctReposition();
  }

  /**
   * Sets the translation and scale to a target point. Like translateScaleToTarget, but instead of taking a scaleDelta
   * it takes the final scale to be used for the target Nodes matrix.
   * @private
   *
   * @param {Vector2} globalPoint - point to translate to in the global coordinate frame
   * @param {number} scale - final scale for the transformation matrix
   */
  setTranslationScaleToTarget( globalPoint, scale ) {
    const pointInLocalFrame = this._targetNode.globalToLocalPoint( globalPoint );
    const pointInParentFrame = this._targetNode.globalToParentPoint( globalPoint );

    const fromLocalPoint = Matrix3.translation( -pointInLocalFrame.x, -pointInLocalFrame.y );
    const toTargetPoint = Matrix3.translation( pointInParentFrame.x, pointInParentFrame.y );

    const nextScale = this.limitScale( scale );

    // we first translate from target point, then apply scale, then translate back to target point ()
    // so that it appears as though we are zooming into that point
    const scaleMatrix = toTargetPoint.timesMatrix( Matrix3.scaling( nextScale ) ).timesMatrix( fromLocalPoint );
    this.matrixProperty.set( scaleMatrix );

    // make sure that we are still within PanZoomListener constraints
    this.correctReposition();
  }

  /**
   * Translate the target node in a direction specified by deltaVector.
   * @public
   *
   * @param {Vector2} deltaVector
   */
  translateDelta( deltaVector ) {
    const targetPoint = this._targetNode.globalToParentPoint( this._panBounds.center );
    const sourcePoint = targetPoint.plus( deltaVector );
    this.translateToTarget( sourcePoint, targetPoint );
  }

  /**
   * Translate the targetNode from a local point to a target point. Both points should be in the global coordinate
   * frame.
   * @public
   *
   * @param {Vector} initialPoint - in global coordinate frame, source position
   * @param {Vector2} targetPoint - in global coordinate frame, target position
   */
  translateToTarget( initialPoint, targetPoint ) {

    const singleInitialPoint = this._targetNode.globalToParentPoint( initialPoint );
    const singleTargetPoint = this._targetNode.globalToParentPoint( targetPoint );
    const delta = singleTargetPoint.minus( singleInitialPoint );
    this.matrixProperty.set( Matrix3.translationFromVector( delta ).timesMatrix( this._targetNode.getMatrix() ) );

    this.correctReposition();
  }

  /**
   * Repositions the target node in response to keyboard input.
   * @private
   *
   * @param   {KeyPress} keyPress
   */
  repositionFromKeys( keyPress ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener reposition from key press' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    const newScale = keyPress.scale;
    const currentScale = this.getCurrentScale();
    if ( newScale !== currentScale ) {

      // key press changed scale
      this.setDestinationScale( newScale );
      this.scaleGestureTargetPosition = keyPress.computeScaleTargetFromKeyPress();
    }
    else if ( !keyPress.translationVector.equals( Vector2.ZERO ) ) {

      // key press initiated some translation
      this.setDestinationPosition( this.sourcePosition.plus( keyPress.translationVector ) );
    }

    this.correctReposition();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Repositions the target node in response to wheel input. Wheel input can come from a mouse, trackpad, or
   * other. Aspects of the event are slightly different for each input source and this function tries to normalize
   * these differences.
   * @private
   *
   * @param   {Wheel} wheel
   * @param   {SceneryEvent} event
   */
  repositionFromWheel( wheel, event ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener reposition from wheel' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // prevent any native browser zoom and don't allow browser go go 'back' or 'forward' a page
    event.domEvent.preventDefault();

    if ( wheel.isCtrlKeyDown ) {
      const nextScale = this.limitScale( this.getCurrentScale() + wheel.scaleDelta );
      this.scaleGestureTargetPosition = wheel.targetPoint;
      this.setDestinationScale( nextScale );
    }
    else {

      // wheel does not indicate zoom, must be translation
      this.setDestinationPosition( this.sourcePosition.plus( wheel.translationVector ) );
    }

    this.correctReposition();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Upon any kind of reposition, update the source position and scale for the next update in animateToTargets.
   *
   * Note: This assumes that any kind of repositioning of the target node will eventually call correctReposition.
   * @protected
   */
  correctReposition() {
    super.correctReposition();

    if ( this._panBounds.isFinite() ) {

      // the pan bounds in the local coordinate frame of the target Node (generally, bounds of the targetNode
      // that are visible in the global panBounds)
      this._transformedPanBounds = this._panBounds.transformed( this._targetNode.matrix.inverted() );

      this.sourcePosition = this._transformedPanBounds.center;
      this.sourceScale = this.getCurrentScale();
    }
  }

  /**
   * When a new press begins, stop any in progress animation.
   * @override
   * @protected
   *
   * @param {MultiListener.Press} press
   */
  addPress( press ) {
    super.addPress( press );
    this.stopInProgressAnimation();
  }

  /**
   * When presses are removed, reset animation destinations.
   * @override
   * @protected
   *
   * @param {MultiListener.Press} press
   * @returns {}
   */
  removePress( press ) {
    super.removePress( press );

    // restore the cursor if we have a middle press as we are in a state where moving the mouse will pan
    if ( this.middlePress ) {
      press.pointer.cursor = MOVE_CURSOR;
    }

    if ( this._presses.length === 0 ) {
      this.stopInProgressAnimation();
    }
  }

  /**
   * Interrupt the listener. Cancels any active input and clears references upon interaction end.
   * @public
   */
  interrupt() {
    this.cancelPanningDuringDrag();

    this.cancelMiddlePress();
    super.interrupt();
  }

  /**
   * "Cancel" the listener, when input stops abnormally. Part of the scenery Input API.
   * @public
   */
  cancel() {
    this.interrupt();
  }

  /**
   * Returns true if the Intent of the Pointer indicates that it will be used for dragging of some kind.
   * @private
   *
   * @param {Pointer} pointer
   * @returns {boolean}
   */
  hasDragIntent( pointer ) {
    return pointer.hasIntent( Intent.KEYBOARD_DRAG ) ||
           pointer.hasIntent( Intent.DRAG );
  }

  /**
   * Pan to a provided Node, attempting to place the node in the center of the transformedPanBounds. It may not end
   * up exactly in the center since we have to make sure panBounds are completely filled with targetNode content.
   * @public
   *
   * @param {Node} node
   */
  panToNode( node ) {
    assert && assert( this._panBounds.isFinite(), 'panBounds should be defined when panning.' );
    this.keepBoundsInView( node.globalBounds, true );
  }

  /**
   * Set the destination position to pan such that the provided globalBounds are totally visible within the panBounds.
   * This will never pan outside panBounds, if the provided globalBounds extend beyond them.
   *
   * If we are not using panToCenter and the globalBounds is larger than the screen size this function does nothing.
   * It doesn't make sense to try to keep the provided bounds entirely in view if they are larger than the availalable
   * view space.
   * @private
   *
   * @param {Bounds2} globalBounds - in global coordinate frame
   * @param {boolean} panToCenter - if true, we will pan to the center of the provided bounds, otherwise we will pan
   *                                until all edges are on screen
   */
  keepBoundsInView( globalBounds, panToCenter = false ) {
    assert && assert( this._panBounds.isFinite(), 'panBounds should be defined when panning.' );

    const boundsInTargetFrame = this._targetNode.globalToLocalBounds( globalBounds );
    const translationDelta = new Vector2( 0, 0 );

    let distanceToLeftEdge = 0;
    let distanceToRightEdge = 0;
    let distanceToTopEdge = 0;
    let distanceToBottomEdge = 0;

    if ( panToCenter ) {

      // If panning to center, the amount to pan is the distance between the center of the screen to the center of the
      // provided bounds. In this case
      distanceToLeftEdge = this._transformedPanBounds.centerX - boundsInTargetFrame.centerX;
      distanceToRightEdge = this._transformedPanBounds.centerX - boundsInTargetFrame.centerX;
      distanceToTopEdge = this._transformedPanBounds.centerY - boundsInTargetFrame.centerY;
      distanceToBottomEdge = this._transformedPanBounds.centerY - boundsInTargetFrame.centerY;
    }
    else if ( boundsInTargetFrame.width < this._transformedPanBounds.width && boundsInTargetFrame.height < this._transformedPanBounds.height ) {

      // If the provided bounds are wider than the available pan bounds we shouldn't try to shift it, it will awkwardly
      // try to slide the screen to one of the sides of the bounds. This operation only makes sense if the screen can
      // totally contain the object being dragged.

      distanceToLeftEdge = this._transformedPanBounds.left - boundsInTargetFrame.left;
      distanceToRightEdge = this._transformedPanBounds.right - boundsInTargetFrame.right;
      distanceToTopEdge = this._transformedPanBounds.top - boundsInTargetFrame.top;
      distanceToBottomEdge = this._transformedPanBounds.bottom - boundsInTargetFrame.bottom;
    }

    if ( distanceToBottomEdge < 0 ) {
      translationDelta.y = -distanceToBottomEdge;
    }
    if ( distanceToTopEdge > 0 ) {
      translationDelta.y = -distanceToTopEdge;
    }
    if ( distanceToRightEdge < 0 ) {
      translationDelta.x = -distanceToRightEdge;
    }
    if ( distanceToLeftEdge > 0 ) {
      translationDelta.x = -distanceToLeftEdge;
    }

    this.setDestinationPosition( this.sourcePosition.plus( translationDelta ) );
  }

  /**
   * Keep a trail in view by panning to it if it has bounds that are outside of the global panBounds.
   * @private
   * @param {Trail} trail
   */
  keepTrailInView( trail ) {
    if ( this._panBounds.isFinite() && trail.lastNode().bounds.isFinite() ) {
      const globalBounds = trail.localToGlobalBounds( trail.lastNode().localBounds );
      if ( !this._panBounds.containsBounds( globalBounds ) ) {
        this.keepBoundsInView( globalBounds, true );
      }
    }
  }

  /**
   * @private
   * @param {number} dt - in seconds
   */
  animateToTargets( dt ) {
    assert && assert( this.boundsFinite, 'initializePositions must be called at least once before animating' );
    assert && assert( this.sourcePosition.isFinite(), 'How can the source position not be a finite Vector2?' );
    assert && assert( this.destinationPosition.isFinite(), 'How can the destination position not be a finite Vector2?' );

    // only animate to targets if within this precision so that we don't animate forever, since animation speed
    // is dependent on the difference betwen source and destination positions
    const positionDirty = !this.destinationPosition.equalsEpsilon( this.sourcePosition, 0.1 );
    const scaleDirty = !Utils.equalsEpsilon( this.sourceScale, this.destinationScale, 0.001 );

    // Only a MiddlePress can support animation while down
    if ( this._presses.length === 0 || this.middlePress !== null ) {
      if ( positionDirty ) {

        // animate to the position, effectively panning over time without any scaling
        const translationDifference = this.destinationPosition.minus( this.sourcePosition );

        let translationDirection = translationDifference;
        if ( translationDifference.magnitude !== 0 ) {
          translationDirection = translationDifference.normalized();
        }

        const translationSpeed = this.getTranslationSpeed( translationDifference.magnitude );
        scratchVelocityVector.setXY( translationSpeed, translationSpeed );

        // finally determine the final panning translation and apply
        const componentMagnitude = scratchVelocityVector.multiplyScalar( dt );
        const translationDelta = translationDirection.componentTimes( componentMagnitude );

        // in case of large dt, don't overshoot the destination
        if ( translationDelta.magnitude > translationDifference.magnitude ) {
          translationDelta.set( translationDifference );
        }

        assert && assert( translationDelta.isFinite(), 'Trying to translate with a non-finite Vector2' );
        this.translateDelta( translationDelta );
      }

      if ( scaleDirty ) {
        assert && assert( this.scaleGestureTargetPosition, 'there must be a scale target point' );

        const scaleDifference = this.destinationScale - this.sourceScale;
        let scaleDelta = scaleDifference * dt * 6;

        // in case of large dt make sure that we don't overshoot our destination
        if ( Math.abs( scaleDelta ) > Math.abs( scaleDifference ) ) {
          scaleDelta = scaleDifference;
        }
        this.translateScaleToTarget( this.scaleGestureTargetPosition, scaleDelta );

        // after applying the scale, the source position has changed, update destination to match
        this.setDestinationPosition( this.sourcePosition );
      }
      else if ( this.destinationScale !== this.sourceScale ) {

        // not far enough to animate but close enough that we can set destination equal to source to avoid further
        // animation steps
        this.setTranslationScaleToTarget( this.scaleGestureTargetPosition, this.destinationScale );
        this.setDestinationPosition( this.sourcePosition );
      }
    }
  }

  /**
   * Stop any in-progress transformations of the target node by setting destinations to sources immediately.
   *
   * @private
   */
  stopInProgressAnimation() {
    if ( this.boundsFinite ) {
      this.setDestinationScale( this.sourceScale );
      this.setDestinationPosition( this.sourcePosition );
    }
  }

  /**
   * Sets the source and destination positions. Necessary because target or pan bounds may not be defined
   * upon construction. This can set those up when they are defined.
   *
   * @private
   */
  initializePositions() {
    this.boundsFinite = this._transformedPanBounds.isFinite();

    if ( this.boundsFinite ) {

      this.sourcePosition = this._transformedPanBounds.center;
      this.setDestinationPosition( this.sourcePosition );
    }
    else {
      this.sourcePosition = null;
      this.destinationPosition = null;
    }
  }

  /**
   * @public
   * @override
   *
   * @param {Bounds2} bounds
   */
  setPanBounds( bounds ) {
    super.setPanBounds( bounds );
    this.initializePositions();

    // drag bounds eroded a bit so that repositioning during drag occurs as the pointer gets close to the edge.
    this._dragBounds = bounds.erodedXY( bounds.width * 0.1, bounds.height * 0.1 );
    assert && assert( this._dragBounds.hasNonzeroArea(), 'drag bounds must have some width and height' );
  }

  /**
   * Upon setting target bounds, re-set source and destination positions.
   * @public
   * @override
   *
   * @param {Bounds2} targetBounds
   */
  setTargetBounds( targetBounds ) {
    super.setTargetBounds( targetBounds );
    this.initializePositions();
  }

  /**
   * Set the destination position. In animation, we will try move the targetNode until sourcePosition matches
   * this point. Destination is in the local coordinate frame of the target node.
   * @private
   *
   * @param {Vector2} destination
   */
  setDestinationPosition( destination ) {
    assert && assert( this.boundsFinite, 'bounds must be finite before setting destination positions' );
    assert && assert( destination.isFinite(), 'provided destination position is not defined' );

    // limit destination position to be within the available bounds pan bounds
    scratchBounds.setMinMax(
      this.sourcePosition.x - this._transformedPanBounds.left - this._panBounds.left,
      this.sourcePosition.y - this._transformedPanBounds.top - this._panBounds.top,
      this.sourcePosition.x + this._panBounds.right - this._transformedPanBounds.right,
      this.sourcePosition.y + this._panBounds.bottom - this._transformedPanBounds.bottom
    );

    this.destinationPosition = scratchBounds.closestPointTo( destination );
  }

  /**
   * Set the destination scale for the target node. In animation, target node will be repositioned until source
   * scale matches destination scale.
   * @private
   *
   * @param {number} scale
   */
  setDestinationScale( scale ) {
    this.destinationScale = this.limitScale( scale );
  }

  /**
   * Calculate the translation speed to animate from our sourcePosition to our targetPosition. Speed goes to zero
   * as the translationDistance gets smaller for smooth animation as we reach our destination position. This returns
   * a speed in the coordinate frame of the parent of this listener's target Node.
   * @private
   *
   * @param {number} translationDistance
   * @returns {number}
   */
  getTranslationSpeed( translationDistance ) {
    assert && assert( translationDistance >= 0, 'distance for getTranslationSpeed should be a non-negative number' );

    // The larger the scale, that faster we want to translate because the distances between source and destination
    // are smaller when zoomed in. Otherwise, speeds will be slower while zoomed in.
    const scaleDistance = translationDistance * this.getCurrentScale();

    // A maximum translation factor applied to distance to determine a reasonable speed, determined by
    // inspection but could be modified. This impacts how long the "tail" of translation is as we animate.
    // While we animate to the destination position we move quickly far away from the destination and slow down
    // as we get closer to the target. Reduce this value to exaggerate that effect and move more slowly as we
    // get closer to the destination position.
    const maxScaleFactor = 5;

    // speed falls away exponentially as we get closer to our destination so that we appear to "slide" to our
    // destination which looks nice, but also prevents us from animating for too long
    const translationSpeed = scaleDistance * ( 1 / ( Math.pow( scaleDistance, 2 ) - Math.pow( maxScaleFactor, 2 ) ) + maxScaleFactor );

    // translationSpeed could be negative or go to infinity due to the behavior of the exponential calculation above.
    // Make sure that the speed is constrained and greater than zero.
    const limitedTranslationSpeed = Math.min( Math.abs( translationSpeed ), MAX_TRANSLATION_SPEED * this.getCurrentScale() );
    return limitedTranslationSpeed;
  }

  /**
   * Reset all transformations on the target node, and reset destination targets to source values to prevent any
   * in progress animation.
   * @public
   * @override
   */
  resetTransform() {
    super.resetTransform();
    this.stopInProgressAnimation();
  }

  /**
   * Get the next discrete scale from the current scale. Will be one of the scales along the discreteScales list
   * and limited by the min and max scales assigned to this MultiPanZoomListener.
   * @private
   *
   * @param {boolean} zoomIn - direction of zoom change, positive if zooming in
   * @returns {number} number
   */
  getNextDiscreteScale( zoomIn ) {

    const currentScale = this.getCurrentScale();

    let nearestIndex;
    let distanceToCurrentScale = Number.POSITIVE_INFINITY;
    for ( let i = 0; i < this.discreteScales.length; i++ ) {
      const distance = Math.abs( this.discreteScales[ i ] - currentScale );
      if ( distance < distanceToCurrentScale ) {
        distanceToCurrentScale = distance;
        nearestIndex = i;
      }
    }

    let nextIndex = zoomIn ? nearestIndex + 1 : nearestIndex - 1;
    nextIndex = Utils.clamp( nextIndex, 0, this.discreteScales.length - 1 );
    return this.discreteScales[ nextIndex ];
  }

  /**
   * @public
   */
  dispose() {
    this.disposeAnimatedPanZoomListener();
  }
}

/**
 * A type that contains the information needed to respond to keyboard input.
 */
class KeyPress {

  /**
   * @param {KeyStateTracker} keyStateTracker
   * @param {KeyStateTracker} scale
   * @param {number} targetScale - scale describing the targetNode, see PanZoomListener._targetScale
   * @param {Object} [options]
   * @returns {KeyStateTracker}
   */
  constructor( keyStateTracker, scale, targetScale, options ) {

    options = merge( {

      // magnitude for translation vector for the target node as long as arrow keys are held down
      translationMagnitude: 80
    }, options );

    // determine resulting translation
    let xDirection = 0;
    xDirection += keyStateTracker.isKeyDown( KeyboardUtils.KEY_RIGHT_ARROW );
    xDirection -= keyStateTracker.isKeyDown( KeyboardUtils.KEY_LEFT_ARROW );

    let yDirection = 0;
    yDirection += keyStateTracker.isKeyDown( KeyboardUtils.KEY_DOWN_ARROW );
    yDirection -= keyStateTracker.isKeyDown( KeyboardUtils.KEY_UP_ARROW );

    // don't set magnitude if zero vector (as vector will become ill-defined)
    scratchTranslationVector.setXY( xDirection, yDirection );
    if ( !scratchTranslationVector.equals( Vector2.ZERO ) ) {
      const translationMagnitude = options.translationMagnitude * targetScale;
      scratchTranslationVector.setMagnitude( translationMagnitude );
    }

    // @public (read-only) - The translation delta vector that should be applied to the target node in response
    // to the key presses
    this.translationVector = scratchTranslationVector;

    // @public (read-only) {number} - determine resulting scale and scale point
    this.scale = scale;
  }

  /**
   * Compute the target position for scaling from a key press. The target node will appear to get larger and zoom
   * into this point. If focus is within the Display, we zoom into the focused node. If not and focusable content
   * exists in the display, we zoom into the first focusable component. Otherwise, we zoom into the top left corner
   * of the screen.
   *
   * This function could be expensive, so we only call it if we know that the key press is a "scale" gesture.
   *
   * @public
   * @returns {Vector2} - a scratch Vector2 instance with the target postion
   */
  computeScaleTargetFromKeyPress() {

    // default cause, scale target will be origin of the screen
    scratchScaleTargetVector.setXY( 0, 0 );

    // zoom into the focused Node if it has defined bounds, it may not if it is for controlling the
    // virtual cursor and has an invisible focus highlight
    const focus = FocusManager.pdomFocusProperty.value;
    if ( focus ) {
      const focusTrail = FocusManager.pdomFocusProperty.value.trail;
      const focusedNode = focusTrail.lastNode();
      if ( focusedNode.bounds.isFinite() ) {
        scratchScaleTargetVector.set( focusTrail.parentToGlobalPoint( focusedNode.center ) );
      }
    }
    else {

      // no focusable element in the Display so try to zoom into the first focusable element
      const firstFocusable = PDOMUtils.getNextFocusable();
      if ( firstFocusable !== document.body ) {

        // if not the body, focused node should be contained by the body - error loudly if the browser reports
        // that this is not the case
        assert && assert( document.body.contains( firstFocusable ), 'focusable should be attached to the body' );

        // assumes that focusable DOM elements are correctly positioned, which should be the case - an alternative
        // could be to use Displat.getTrailFromPDOMIndicesString(), but that function requires information that is not
        // available here.
        const centerX = firstFocusable.offsetLeft + firstFocusable.offsetWidth / 2;
        const centerY = firstFocusable.offsetTop + firstFocusable.offsetHeight / 2;
        scratchScaleTargetVector.setXY( centerX, centerY );
      }
    }

    assert && assert( scratchScaleTargetVector.isFinite(), 'target position not defined' );
    return scratchScaleTargetVector;
  }
}

/**
 * A type that contains the information needed to respond to a wheel input.
 */
class Wheel {

  /**
   * @param {SceneryEvent} event
   * @param {number} targetScale - scale describing the targetNode, see PanZoomListener._targetScale
   */
  constructor( event, targetScale ) {
    const domEvent = event.domEvent;

    // @public (read-only) - is the ctrl key down during this wheel input? Cannot use KeyStateTracker because the
    // ctrl key might be 'down' on this event without going through the keyboard. For example, with a trackpad
    // the browser sets ctrlKey true with the zoom gesture.
    this.isCtrlKeyDown = event.domEvent.ctrlKey;

    // @public (read-only) - magnitude and direction of scale change from the wheel input
    this.scaleDelta = domEvent.deltaY > 0 ? -0.5 : 0.5;

    // @public (read-only) - the target of the wheel input in the global coordinate frame
    this.targetPoint = event.pointer.point;

    // the DOM Event specifies deltas that look appropriate and works well in different cases like
    // mouse wheel and trackpad input, both which trigger wheel events but at different rates with different
    // delta values - but they are generally too large, reducing a bit feels more natural and gives more control
    let translationX = domEvent.deltaX * 0.5;
    let translationY = domEvent.deltaY * 0.5;

    // FireFox defaults to scrolling in units of "lines" rather than pixels, resulting in slow movement - speed up
    // translation in this case
    if ( domEvent.deltaMode === window.WheelEvent.DOM_DELTA_LINE ) {
      translationX = translationX * 25;
      translationY = translationY * 25;
    }

    // @public (read-only)
    this.translationVector = scratchTranslationVector.setXY( translationX * targetScale, translationY * targetScale );
  }
}

/**
 * A press from a middle mouse button. Will initiate panning and destination position will be updated for as long
 * as the Pointer point is dragged away from the initial point.
 */
class MiddlePress {

  /**
   * @param {Mouse} pointer
   * @param {Trail} trail
   */
  constructor( pointer, trail ) {
    assert && assert( pointer.type === 'mouse', 'incorrect pointer type' );

    // @private
    this.pointer = pointer;
    this.trail = trail;

    // point of press in the global coordinate frame
    this.initialPoint = pointer.point.copy();
  }
}

/**
 * Helper function, calculates discrete scales between min and max scale limits. Creates increasing step sizes
 * so that you zoom in from high zoom reaches the max faster with fewer key presses. This is standard behavior for
 * browser zoom.
 *
 * @param {number} minScale
 * @param {number} maxScale
 * @returns {Array.<number>}
 */
const calculateDiscreteScales = ( minScale, maxScale ) => {

  assert && assert( minScale >= 1, 'min scales less than one are currently not supported' );

  // will take this many key presses to reach maximum scale from minimum scale
  const steps = 8;

  // break the range from min to max scale into steps, then exponentiate
  const discreteScales = [];
  for ( let i = 0; i < steps; i++ ) {
    discreteScales[ i ] = ( maxScale - minScale ) / steps * ( i * i );
  }

  // normalize steps back into range of the min and max scale for this listener
  const discreteScalesMax = discreteScales[ steps - 1 ];
  for ( let i = 0; i < discreteScales.length; i++ ) {
    discreteScales[ i ] = minScale + discreteScales[ i ] * ( maxScale - minScale ) / discreteScalesMax;
  }

  return discreteScales;
};

scenery.register( 'AnimatedPanZoomListener', AnimatedPanZoomListener );
export default AnimatedPanZoomListener;
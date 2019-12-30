// Copyright 2019, University of Colorado Boulder

/**
 * A PanZoomListener that supports additional forms of input for pan and zoom, including trackpad gestures, mouse
 * wheel, and keyboard input. These gestures will animate the target node to its destination translation and scale so it
 * uses a step function that must be called every animation frame.
 *
 * TODO: Implement dispose
 *
 * @author Jesse Greenberg
 */
define( require => {
  'use strict';

  // modules
  const AccessibilityUtils = require( 'SCENERY/accessibility/AccessibilityUtils' );
  const Display = require( 'SCENERY/display/Display' );
  const KeyboardUtils = require( 'SCENERY/accessibility/KeyboardUtils' );
  const KeyboardZoomUtil = require( 'SCENERY/accessibility/KeyboardZoomUtil' );
  const KeyStateTracker = require( 'SCENERY/accessibility/KeyStateTracker' );
  const Matrix3 = require( 'DOT/Matrix3' );
  const merge = require( 'PHET_CORE/merge' );
  const PanZoomListener = require( 'SCENERY/listeners/PanZoomListener' );
  const platform = require( 'PHET_CORE/platform' );
  const Pointer = require( 'SCENERY/input/Pointer' );
  const scenery = require( 'SCENERY/scenery' );
  const Utils = require( 'DOT/Utils' );
  const Vector2 = require( 'DOT/Vector2' );

  // constants
  const MOVE_CURSOR = 'all-scroll';
  const MAX_SCROLL_VELOCITY = 300; // max global view coords per second while scrolling with middle mouse button drag

  // scratch variables to reduce garbage
  const scratchTranslationVector = new Vector2( 0, 0 );
  const scratchScaleTargetVector = new Vector2( 0, 0 );
  const scratchVelocityVector = new Vector2( 0, 0 );

  class AnimatedPanZoomListener extends PanZoomListener {

    /**
     * @param {Node} targetNode - Node to be transformed by this listener
     * @param {KeyStateTracker} keyStateTracker - represents keyboard state to respond to keyboard input
     * @param {Object} options
     */
    constructor( targetNode, keyStateTracker, options ) {
      super( targetNode, options );

      // @private {KeyStateTracker}
      // this.keyStateTracker = keyStateTracker;
      this.keyStateTracker = new KeyStateTracker();
      this.keyStateTracker.attachToBody();

      // @private (null|Vector2) - This point is the center of the transformedPanBounds (see PanZoomListener) in
      // the parent coordinate frame of the targetNode. This is the current center of the transformedPanBounds, and
      // during animation we will move this point closer to the destinationLocation.
      this.sourceLocation = null;

      // @private (null|Vector2) - The destination for translation, we will reposition the targetNode until the
      // sourceLocation matches this point. This is in the parent coordinate frame of the targetNode.
      this.destinationLocation = null;

      // @private {number} - The current scale of the targetNode. During animation we will scale the targetNode until
      // this matches the destinationScale.
      this.sourceScale = this.getCurrentScale();

      // @private {number} - The desired scale for the targetNode, the node is repositioned until sourceScale matches
      // destinationScale.
      this.destinationScale = this.getCurrentScale();

      // @private {null|Vector2} - The point at which a scale gesture was initiated. This is usually the mouse point in
      // the global coordinate frame when a wheel or trackpad zoom gesture is initiated. The targetNode will appear to
      // be zoomed into this point. This is in the global coordinate frame.
      this.scaleGestureTargetLocation = null;

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

      // @private {Vector2|null} - A Pointer point during a drag with another listener which declares its intent
      // for dragging. In the global coordinate frame. We will reposition based on how far this point is out of
      // this._dragBounds (when defined)
      this._repositionDuringDragPoint = null;

      // @private - whether or not the Pointer went down within the drag bounds - if it went down out of drag bounds
      // then user likely trying to pull an object back into view so we prevent panning during drag
      this._downInDragBounds = false;

      // respond to macOS trackpad input, but don't respond to this input on an iOS touch screen
      if ( platform.safari && !platform.mobileSafari ) {

        // @private {number} - the scale of the targetNode at the start of the gesture, used to calculate
        // how scale to apply from 'gesturechange' event
        this.trackpadGestureStartScale = this.getCurrentScale();

        // WARNING: These events are non-standard, but this is the only way to detect and prevent native trackpad
        // input on macOS Safari. For Apple documentation about these events, see
        // https://developer.apple.com/documentation/webkitjs/gestureevent
        // NOTE: Perhaps these can be added to the scenery Input API so we don't have to go through window
        window.addEventListener( 'gesturestart', this.handleGestureStartEvent.bind( this ) );
        window.addEventListener( 'gesturechange', this.handleGestureChangeEvent.bind( this ) );
      }

      // Handle key input from events outside of the PDOM - in this case it is impossible for the a11y pointer
      // to be attached so we have free reign over the keyboard
      this.keyStateTracker.keydownEmitter.addListener( this.documentKeydown.bind( this ) );

      // TODO: move this out of this listener? PanZoomListener shouldn't care about Display focus
      Display.focusProperty.link( focus => {
        if ( focus ) {
          const node = focus.trail.lastNode();
          if ( !this._panBounds.containsBounds( node.globalBounds ) ) {
            this.panToNode( node );
          }
        }
      } );
    }

    /**
     * Step the listener, supporting any animation as the target node is transformed to target location and scale.
     * @public
     *
     * @param {number} dt
     */
    step( dt ) {
      if ( this.middlePress ) {
        this.handleMiddlePress( dt );
      }

      if ( this._repositionDuringDragPoint ) {
        this.repositionDuringDrag( this._repositionDuringDragPoint );
      }

      this.animateToTargets( dt );
    }

    /**
     * Attach a MiddlePress for drag panning, if detected.
     * @override
     *
     * @param {SceneryEvent} event
     */
    down( event ) {
      PanZoomListener.prototype.down.call( this, event );
      this._downTarget = event.target;
      this._downInDragBounds = this._dragBounds.containsPoint( event.pointer.point );

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
      }
    }

    /**
     * Listener for the attached pointer on move. Only move if a middle press is not currently down.
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
     *
     * @param {SceneryEvent} event
     */
    move( event ) {
      if ( this._downTarget && this._downInDragBounds ) {
        const targetInBounds = this._panBounds.containsBounds( this._downTarget.globalBounds );
        const hasDragIntent = this.hasDragIntent( event.pointer );
        const currentTargetExists = event.currentTarget !== null;

        if ( currentTargetExists && hasDragIntent && !targetInBounds ) {
          this._repositionDuringDragPoint = event.pointer.point;
        }
        else {
          this._repositionDuringDragPoint = null;
        }
      }
      else {
        this._downInDragBounds = this._dragBounds.containsPoint( event.pointer.point );
      }
    }

    /**
     * During a drag of another Node that is a descendant of this listener's targetNode, reposition if the
     * node is out of dragBounds so that the Node is always within panBounds.
     * @private
     *
     * @param {Vector2} pointerPoint - in the global coordinate frame
     */
    repositionDuringDrag( pointerPoint ) {
      const closestContainedPoint = this._dragBounds.getClosestPoint( pointerPoint.x, pointerPoint.y );
      const translationDelta = pointerPoint.minus( closestContainedPoint );
      this.setDestinationLocation( this.sourceLocation.plus( translationDelta ) );
    }

    /**
     * Scenery listener API. Clear cursor and middlePress.
     *
     * @param {SceneryEvent} event
     */
    up( event ) {
      this._targetInBoundsOnDown = false;
      this._downTarget = null;
      this._repositionDuringDragPoint = null;
    }

    /**
     * Input listener for the 'wheel' event, part of the Scenery Input API.
     *
     * @param {SceneryEvent} event
     */
    wheel( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener wheel' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // cannot reposition if a dragging with middle mouse button - but wheel zoom should not cancel a middle press
      // (behavior copied from other browsers)
      if ( !this.middlePress ) {
        const wheel = new Wheel( event );
        this.repositionFromWheel( wheel, event );
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }

    /**
     * Keydown listener for events outside of the PDOM. Attached as a listener to the body and driven by
     * Events rather than SceneryEvents. When we handle Events from within the PDOM we need the Pointer to
     * determine if attached. But from outside of the PDOM we know that there is no focus in the document and therfore
     * the A11yPointer is not attached.
     * @private
     *
     * @param {Event} domEvent
     */
    documentKeydown( domEvent ) {

      // on any keyboard reposition interrupt the middle press panning
      this.cancelMiddlePress();

      const displayAccessible = phet.joist.sim.display._accessible;
      if ( !displayAccessible || !phet.joist.sim.display.accessibleDOMElement.contains( domEvent.target ) ) {
        this.handleZoomCommands( domEvent );

        // handle translation without worry of the pointer being attached because there is no pointer at this level
        if ( KeyboardUtils.isArrowKey( domEvent.keyCode ) ) {
          const keyPress = new KeyPress( this.keyStateTracker, this.getCurrentScale() );
          this.repositionFromKeys( keyPress );
        }

      }
    }

    /**
     * For the Scenery listener API, handle a keydown event. This SceneryEvent will have been dispatched from
     * Input.dispatchEvent and so the Event target must be within the PDOM. In this case, we may
     * need to prevent translation if the A11yPointer is attached or the Pointer indicates that it
     * is intended to for arrow key control.
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

      // handle translation
      if ( KeyboardUtils.isArrowKey( domEvent.keyCode ) ) {
        const keyboardDragIntent = event.pointer.getIntent() === Pointer.Intent.KEYBOARD_DRAG;
        // const elementUsesKeys = AccessibilityUtils.elementUsesArrowKeys( domEvent.target );

        if ( !keyboardDragIntent ) {
          sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener handle arrow key down' );
          sceneryLog && sceneryLog.InputListener && sceneryLog.push();

          const keyPress = new KeyPress( this.keyStateTracker, this.getCurrentScale() );
          this.repositionFromKeys( keyPress );

          sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
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
      const zoomInCommandDown = KeyboardZoomUtil.isZoomCommand( domEvent, true );
      const zoomOutCommandDown = KeyboardZoomUtil.isZoomCommand( domEvent, false );

      if ( zoomInCommandDown || zoomOutCommandDown ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiPanZoomListener keyboard zoom in' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        // don't allow native browser zoom
        domEvent.preventDefault();

        const nextScale = this.getNextDiscreteScale( zoomInCommandDown );
        const keyPress = new KeyPress( this.keyStateTracker, nextScale );
        this.repositionFromKeys( keyPress );
      }
      else if ( KeyboardZoomUtil.isZoomResetCommand( domEvent ) ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener keyboard reset' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        // this is a native command, but we are taking over
        event.preventDefault();
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

      // prevent Safari from doing anything native with this gesture
      domEvent.preventDefault();
      this.trackpadGestureStartScale = domEvent.scale;
      this.scaleGestureTargetLocation = new Vector2( event.pageX, event.pageY );
    }

    /**
     * This is just for macOS Safari. Responds to trackpad input. Prevends default browser behavior and
     * sets destination scale as user pinches on the trackpad.
     * @private
     *
     * @param {Event} domEvent
     */
    handleGestureChangeEvent( domEvent ) {

      // prevent Safari from changing position or scale natively
      domEvent.preventDefault();

      const newScale = this.sourceScale + domEvent.scale - this.trackpadGestureStartScale;
      this.setDestinationScale( newScale );
    }

    /**
     * Handle the down MiddlePress during animation. If we have a middle press we need to update location target.
     * @private
     *
     * @param {number} dt
     */
    handleMiddlePress( dt ) {
      assert && assert( this.middlePress, 'MiddlePress must be defined to handle' );

      const currentPoint = this.middlePress.pointer.point;
      const globalDelta = currentPoint.minus( this.middlePress.initialPoint );

      // magnitude alone is too fast, reduce by a bit
      const reducedMagnitude = globalDelta.magnitude / 100;
      if ( dt > 0 && reducedMagnitude > 0 ) {

        // set the delta vector in global coordinates, limited by a maximum view coords/second velocity
        globalDelta.setMagnitude( Math.min( reducedMagnitude / dt, MAX_SCROLL_VELOCITY ) );
        this.setDestinationLocation( this.sourceLocation.plus( globalDelta ) );
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

      var fromLocalPoint = Matrix3.translation( -pointInLocalFrame.x, -pointInLocalFrame.y );
      var toTargetPoint = Matrix3.translation( pointInParentFrame.x, pointInParentFrame.y );

      const nextScale = this.limitScale( this.getCurrentScale() + scaleDelta );

      // we first translate from target point, then apply scale, then translate back to target point ()
      // so that it appears as though we are zooming into that point
      const scaleMatrix = toTargetPoint.timesMatrix( Matrix3.scaling( nextScale ) ).timesMatrix( fromLocalPoint );
      this._targetNode.matrix = scaleMatrix;

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

      // TODO: scratch things?
      const singleInitialPoint = this._targetNode.globalToParentPoint( initialPoint );
      const singleTargetPoint = this._targetNode.globalToParentPoint( targetPoint );
      var delta = singleTargetPoint.minus( singleInitialPoint );
      this._targetNode.matrix = Matrix3.translationFromVector( delta ).timesMatrix( this._targetNode.getMatrix() );

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
        this.scaleGestureTargetLocation = keyPress.computeScaleTargetFromKeyPress();
      }
      else if ( !keyPress.translationVector.equals( Vector2.ZERO ) ) {

        // key press initiated some translation
        this.setDestinationLocation( this.sourceLocation.plus( keyPress.translationVector ) );
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
        this.scaleGestureTargetLocation = wheel.targetPoint;
        this.setDestinationScale( nextScale );
      }
      else {

        // wheel does not indicate zoom, must be translation
        this.setDestinationLocation( this.sourceLocation.plus( wheel.translationVector ) );
      }

      this.correctReposition();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }

    /**
     * Upon any kind of reposition, update the source location and scale for the next update in animateToTargets.
     *
     * Note: This assumes that any kind of repositioning of the target node will eventually call correctReposition.
     */
    correctReposition() {
      super.correctReposition();

      this.sourceLocation = this._transformedPanBounds.center;
      this.sourceScale = this.getCurrentScale();
    }

    /**
     * When a new press begins, stop any in progress animation.
     *
     * @param {MultiListener.Press} press
     */
    addPress( press ) {
      super.addPress( press );
      this.stopInProgressAnimation();
    }

    /**
     * When presses are removed, reset animation destinations.
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
     * Interrupt the listener.
     * @public
     */
    interrupt() {
      this.cancelMiddlePress();
      super.interrupt();
    }

    hasDragIntent( pointer ) {
      return pointer.getIntent() === Pointer.Intent.KEYBOARD_DRAG ||
             pointer.getIntent() === Pointer.Intent.DRAG ||
             pointer.getIntent() === Pointer.Intent.MULTI_DRAG;
    }

    /**
     * Pan to a provided Node, attempting to place the node in the center of the transformedPanBounds. It may not end
     * up exactly in the center since we have to make sure panBounds are completely filled with targetNode content.
     * @public
     *
     * @param {Node} node
     */
    panToNode( node ) {
      const globalLocation = node.globalBounds.center;

      const locationInTargetFrame = this._targetNode.globalToLocalPoint( globalLocation );

      const distanceToLeftEdge = Math.abs( this._targetBounds.left - locationInTargetFrame.x );
      const distanceToRightEdge = Math.abs( this._targetBounds.right - locationInTargetFrame.x );
      const distanceToTopEdge = Math.abs( this._targetBounds.top - locationInTargetFrame.y );
      const distanceToBottomEdge = Math.abs( this._targetBounds.bottom - locationInTargetFrame.y );

      if ( distanceToRightEdge < this._transformedPanBounds.width / 2  ) {
        const correction = this._transformedPanBounds.width / 2 - distanceToRightEdge;
        locationInTargetFrame.x = locationInTargetFrame.x - correction;
      }
      if ( distanceToLeftEdge < this._transformedPanBounds.width / 2 ) {
        const correction = this._transformedPanBounds.width / 2 - distanceToLeftEdge;
        locationInTargetFrame.x = locationInTargetFrame.x + correction;
      }
      if ( distanceToTopEdge < this._transformedPanBounds.height / 2 ) {
        const correction = this._transformedPanBounds.height / 2 - distanceToTopEdge;
        locationInTargetFrame.y = locationInTargetFrame.y + correction;
      }
      if ( distanceToBottomEdge < this._transformedPanBounds.height / 2 ) {
        const correction = this._transformedPanBounds.height / 2 - distanceToBottomEdge;
        locationInTargetFrame.y = locationInTargetFrame.y - correction;
      }

      this.setDestinationLocation( locationInTargetFrame );
    }

    animateToTargets( dt ) {
      assert && assert( this.destinationLocation !== null, 'initializeLocations must be called at least once before animating' );
      assert && assert( this.sourceLocation !== null, 'initializeLocations must be called at least once before animating' );

      const locationDirty = !this.destinationLocation.equalsEpsilon( this.sourceLocation, 0.1 );
      const scaleDirty = !Utils.equalsEpsilon( this.sourceScale, this.destinationScale, 0.001 );

      // Only a MiddlePress can support animation while down
      if ( this._presses.length === 0 || this.middlePress !== null ) {
        if ( locationDirty ) {

          // animate to the location, effectively panning over time without any scaling
          const translationDifference = this.destinationLocation.minus( this.sourceLocation );

          let translationDirection = translationDifference;
          if ( translationDifference.magnitude !== 0 ) {
            translationDirection = translationDifference.normalized();
          }

          // translation velocity is faster the farther away you are from the target
          const translationSpeed = translationDifference.magnitude * 6;
          scratchVelocityVector.setXY( translationSpeed, translationSpeed );

          // finally determine the final panning translation and apply
          const componentMagnitude = scratchVelocityVector.multiplyScalar( dt );
          const translationDelta = translationDirection.componentTimes( componentMagnitude );

          // in case of large dt, don't overshoot the destination
          if ( translationDelta.magnitude > translationDifference.magnitude ) {
            translationDelta.set( translationDifference );
          }

          this.translateDelta( translationDelta );
        }
        if ( scaleDirty ) {
          assert && assert( this.scaleGestureTargetLocation, 'there must be a scale target point' );

          const scaleDifference = this.destinationScale - this.sourceScale;
          let scaleDelta = scaleDifference * dt * 6;

          // in case of large dt make sure that we don't overshoot our destination
          if ( Math.abs( scaleDelta ) > Math.abs( scaleDifference ) ) {
            scaleDelta = scaleDifference;
          }
          this.translateScaleToTarget( this.scaleGestureTargetLocation, scaleDelta );

          // after applying the scale, the source position has changed, update destination to match
          this.setDestinationLocation( this.sourceLocation );
        }
      }
    }

    /**
     * Stop any in-progress transformations of the target node by setting destinations to sources immediately.
     *
     * @private
     */
    stopInProgressAnimation() {
      this.setDestinationScale( this.sourceScale );
      this.setDestinationLocation( this.sourceLocation );
    }

    /**
     * Sets the source and destination locations. Necessary because target or pan bounds may not be defined
     * upon construction. This can set those up when they are defined.
     *
     * @private
     */
    initializeLocations() {
      this.sourceLocation = this._transformedPanBounds.center;
      this.setDestinationLocation( this.sourceLocation );
    }

    /**
     * @override
     *
     * @param {Bounds2} bounds
     */
    setPanBounds( bounds ) {
      super.setPanBounds( bounds );
      this.initializeLocations();

      // drag bounds eroded a bit so that repositioning during drag occurs as the pointer gets close to the edge.
      this._dragBounds = bounds.erodedXY( bounds.width * 0.1, bounds.height * 0.1 );
      assert && assert( this._dragBounds.hasNonzeroArea(), 'drag bounds must have some width and height' );
    }

    /**
     * Upon setting target bounds, re-set source and destination locations.
     * @override
     *
     * @param {Bounds2} targetBounds
     */
    setTargetBounds( targetBounds ) {
      super.setTargetBounds( targetBounds );
      this.initializeLocations();
    }

    /**
     * Set the destination location. In animation, we will try move the targetNode until sourceLocation matches
     * this point. Destination is in the local coordinate frame of the target node.
     * @private
     *
     * @param {Vector2} destination
     */
    setDestinationLocation( destination ) {
      this.destinationLocation = destination;
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
     * Reset all transformations on the target node, and reset destination targets to source values to prevent any
     * in progress animation.
     * @override
     */
    resetTransform() {
      super.resetTransform();
      this.stopInProgressAnimation();
    }

    /**
     * Get the next discrete scale from the current scale. Will be one of the scales along the discreteScales list
     * and limited by the min and max scales assigned to this MultiPanZoomListener.
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
     * Returns true if the provided trail has any listeners that use keydown or keyup. If we find such a listener
     * we want to prevent panning with a keyboard. Excludes this listener in the search, and stops searching once we
     * hit it.
     *
     * @param {Trail} trail
     * @returns {boolean}
     */
    hasKeyListeners( trail ) {
      let hasKeyListeners = false;
      let foundThisListener = false;

      // search backwards because it is most likely that nodes adjacent to the focus have a keydown/keyup listener,
      // and so we can stop searching when we find this MultiListener
      for ( let i = trail.length - 1; i >= 0; i-- ) {
        const node = trail.nodes[ i ];
        hasKeyListeners = _.some( node.inputListeners, listener => {
          if ( !foundThisListener && listener === this) {
            foundThisListener = true;
          }
          const hasListeners = _.intersection( _.keys( listener ), [ 'keydown', 'keyup' ] ).length > 0;

          return ( !foundThisListener && hasListeners );
        } );

        // don't keep searching if we find this listener or any with the above listeners
        if ( hasKeyListeners || foundThisListener ) { break; }
      }

      return hasKeyListeners;
    }
  }

  /**
   * A type that contains the information needed to respond to keyboard input.
   */
  class KeyPress {

    /**
     * @param {KeyStateTracker} keyStateTracker
     * @param {KeyStateTracker} scale
     * @param {KeyStateTracker} options
     * @returns {KeyStateTracker}
     */
    constructor( keyStateTracker, scale, options ) {

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
        scratchTranslationVector.setMagnitude( options.translationMagnitude );
      }

      // @public (read-only) - The translation delta vector that should be applied to the target node in response
      // to the key presses
      this.translationVector = scratchTranslationVector;

      // @public (read-only) {number} - determine resulting scale and scale point
      this.scale = scale;
    }

    /**
     * Compute the target location for scaling from a key press. The target node will appear to get larger and zoom
     * into this point. If focus is within the Display, we zoom into the focused node. If not and focusable content
     * exists in the display, we zoom into the first focusable component. Otherwise, we zoom into the top left corner
     * of the screen.
     *
     * This function could be expensive, so we only call it if we know that the key press is a "scale" gesture.
     *
     * TODO: Consider moving somewhere else, it seems that MultiPanZoomListener shouldn't care about Display's focus.
     * And this behavior assumes that the use of this will be at the global level. Maybe we need a SimPanZoomListener
     * that extends MultiPanZoomListener.
     *
     * @public
     * @returns {}
     */
    computeScaleTargetFromKeyPress() {

      // default cause, scale target will be origin of the screen
      scratchScaleTargetVector.setXY( 0, 0 );

      const focusedNode = scenery.Display.focusedNode;
      if ( focusedNode ) {
        scratchScaleTargetVector.set( focusedNode.parentToGlobalPoint( focusedNode.center ) );
      }
      else {
        const firstFocusable = AccessibilityUtils.getNextFocusable();
        if ( firstFocusable !== document.body ) {

          // if not the body, focused node should be contained by the body - error loudly if the browser reports
          // that this is not the case
          assert && assert( document.body.contains( firstFocusable ), 'focusable should be attached to the body' );

          // assumes that focusable DOM elements are correctly positioned, which should be the case - an alternative
          // could be to use Trail.fromUniqueId, but that function requires information that is not available here
          const centerX = firstFocusable.offsetLeft + firstFocusable.offsetWidth / 2;
          const centerY = firstFocusable.offsetTop + firstFocusable.offsetHeight / 2;
          scratchScaleTargetVector.setXY( centerX, centerY );
        }
      }

      return scratchScaleTargetVector;
    }
  }

  /**
   * A type that contains the information needed to respond to a wheel input.
   */
  class Wheel {

    /**
     * @param {SceneryEvent} event
     */
    constructor( event ) {
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
      // delta values
      let translationX = domEvent.deltaX;
      let translationY = domEvent.deltaY;

      // FireFox defaults to scrolling in units of "lines" rather than pixels, resulting in slow movement - speed up
      // translation in this case
      if ( domEvent.deltaMode === window.WheelEvent.DOM_DELTA_LINE ) {
        translationX = translationX * 25;
        translationY = translationY * 25;
      }

      // @public (read-only)
      this.translationVector = scratchTranslationVector.setXY( translationX, translationY );
    }
  }

  /**
   * A press from a middle mouse button. Will initiate panning and destination location will be updated for as long
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
      discreteScales[ i ] = ( maxScale - minScale ) / steps * ( i * i  );
    }

    // normalize steps back into range of the min and max scale for this listener
    const discreteScalesMax = discreteScales[ steps - 1 ];
    for ( let i = 0; i < discreteScales.length; i++ ) {
      discreteScales[ i ] = minScale + discreteScales[ i ] * ( maxScale - minScale ) / discreteScalesMax;
    }

    return discreteScales;
  };

  return scenery.register( 'AnimatedPanZoomListener', AnimatedPanZoomListener );
} );

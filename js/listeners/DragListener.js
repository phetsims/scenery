// Copyright 2017, University of Colorado Boulder

/**
 * PressListener subtype customized for handling most drag-related listener needs.
 *
 * DragListener uses some specific terminology that is helpful to understand:
 *
 * - Drag target: The node whose trail is used for coordinate transforms. When a targetNode is specified, it will be the
 *                drag target. Otherwise, whatever was the currentTarget during event bubbling for the event that
 *                triggered press will be used (almost always the node that the listener is added to).
 * - Global coordinate frame: Coordinate frame of the Display (specifically its rootNode's local coordinate frame),
 *                            that in some applications will be screen coordinates.
 * - Parent coordinate frame: The parent coordinate frame of our drag target. Basically, it's the coordinate frame
 *                            you'd need to use to set dragTarget.translation = <parent coordinate frame point> for the
 *                            drag target to follow the pointer.
 * - Local coordinate frame: The local coordinate frame of our drag target, where (0,0) would be at the drag target's
 *                           origin.
 * - Model coordinate frame: Optionally defined by a model-view transform (treating the parent coordinate frame as the
 *                           view). When a transform is provided, it's the coordinate frame needed for setting
 *                           dragModelElement.position = <model coordinate frame point>. If a transform is not provided
 *                           (or overridden), it will be the same as the parent coordinate frame.
 *
 * The typical coordinate handling of DragListener is to:
 * 1. When a drag is started (with press), record the pointer's location in the local coordinate frame. This is visually
 *    where the pointer is over the drag target, and typically most drags will want to move the dragged element so that
 *    the pointer continues to be over this point.
 * 2. When the pointer is moved, compute the new parent translation to keep the pointer on the same place on the
 *    dragged element.
 * 3. (optionally) map that to a model location, and (optionally) move that model location to satisfy any constraints of
 *    where the element can be dragged (recomputing the parent/model translation as needed)
 * 4. Apply the required translation (with a provided drag callback, using the locationProperty, or directly
 *    transforming the Node if translateNode:true).
 *
 * For example usage, see scenery/examples/input.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  // modules
  var Action = require( 'AXON/Action' );
  var ActionIO = require( 'AXON/ActionIO' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var EventIO = require( 'SCENERY/input/EventIO' );
  var inherit = require( 'PHET_CORE/inherit' );
  var PhetioObject = require( 'TANDEM/PhetioObject' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );
  var Tandem = require( 'TANDEM/Tandem' );
  var Touch = require( 'SCENERY/input/Touch' );
  var Transform3 = require( 'DOT/Transform3' );
  var TransformTracker = require( 'SCENERY/util/TransformTracker' );
  var Vector2 = require( 'DOT/Vector2' );

  // Scratch vectors used to prevent allocations
  var scratchVector2A = new Vector2( 0, 0 );
  var DraggedActionIO = ActionIO( [ { name: 'event', type: EventIO } ] );

  /**
   * @constructor
   * @extends PressListener
   *
   * @param {Object} [options] - See the constructor body (below) and in PressListener for documented options.
   */
  function DragListener( options ) {
    var self = this;

    options = _.extend( {
      // {boolean} - If true, unattached touches that move across our node will trigger a press(). This helps sometimes
      // for small draggable objects.
      allowTouchSnag: true,

      // {boolean} - If true, the initial offset of the pointer's location is taken into account, so that drags will
      // try to keep the pointer at the same local point of our dragged node.
      applyOffset: true,

      // {boolean} - If true, ancestor transforms will be watched. If they change, it will trigger a repositioning,
      // which will usually adjust the location/transform to maintain position.
      trackAncestors: false,

      // {boolean} - If true, the effective currentTarget will be translated when the drag position changes.
      translateNode: false,

      // {Transform3|null} - If provided, this will be the conversion between the parent (view) and model coordinate
      // frames. Usually most useful when paired with the locationProperty.
      transform: null,

      // {Property.<Vector2>|null} - If provided, it will be synchronized with the drag location in the model coordinate
      // frame (applying any provided transforms as needed).
      locationProperty: null,

      // {Property.<Bounds2|null>} - If provided, the model location will be constrained to be inside these bounds.
      dragBoundsProperty: null,

      // {Function|null} - function( modelPoint: {Vector2} ) : {Vector2}. If provided, it will allow custom mapping
      // from the desired location (i.e. where the pointer is) to the actual possible location (i.e. where the dragged
      // object ends up). For example, using dragBoundsProperty is equivalent to passing:
      //   mapLocation: function( point ) { return dragBoundsProperty.value.closestPointTo( point ); }
      mapLocation: null,

      // {Function|null} - function( viewPoint: {Vector2}, listener: {DragListener} ) : {Vector2}. If provided, its
      // result will be added to the parentPoint before computation continues, to allow the ability to "offset" where
      // the pointer location seems to be. Useful for touch, where things shouldn't be under the pointer directly.
      offsetLocation: null,

      // {Function|null} - Called as start( event: {Event}, listener: {DragListener} ) when the drag is started.
      // This is preferred over passing press(), as the drag start hasn't been fully processed at that point.
      start: null,

      // {Function|null} - Called as end( listener: {DragListener} ) when the drag is ended. This is preferred over
      // passing release(), as the drag start hasn't been fully processed at that point.
      end: null,

      // {Tandem} - For instrumenting
      tandem: Tandem.required,

      // to support properly passing this to children, see https://github.com/phetsims/tandem/issues/60
      phetioReadOnly: PhetioObject.DEFAULT_OPTIONS.phetioReadOnly,
      phetioFeatured: PhetioObject.DEFAULT_OPTIONS.phetioFeatured
    }, options );

    assert && assert( typeof options.allowTouchSnag === 'boolean', 'allowTouchSnag should be a boolean' );
    assert && assert( typeof options.applyOffset === 'boolean', 'applyOffset should be a boolean' );
    assert && assert( typeof options.trackAncestors === 'boolean', 'trackAncestors should be a boolean' );
    assert && assert( typeof options.translateNode === 'boolean', 'translateNode should be a boolean' );
    assert && assert( options.transform === null || options.transform instanceof Transform3, 'transform, if provided, should be a Transform3' );
    assert && assert( options.locationProperty === null || options.locationProperty instanceof Property, 'locationProperty, if provided, should be a Property' );
    assert && assert( !options.dragBounds, 'options.dragBounds was removed in favor of options.dragBoundsProperty' );
    assert && assert( options.dragBoundsProperty === null || options.dragBoundsProperty instanceof Property, 'dragBoundsProperty, if provided, should be a Property' );
    assert && assert( options.mapLocation === null || typeof options.mapLocation === 'function', 'mapLocation, if provided, should be a function' );
    assert && assert( options.offsetLocation === null || typeof options.offsetLocation === 'function', 'offsetLocation, if provided, should be a function' );
    assert && assert( options.start === null || typeof options.start === 'function', 'start, if provided, should be a function' );
    assert && assert( options.end === null || typeof options.end === 'function', 'end, if provided, should be a function' );
    assert && assert( options.tandem instanceof Tandem, 'The provided tandem should be a Tandem' );

    assert && assert(
      !( options.mapLocation && options.dragBoundsProperty ),
      'Only one of mapLocation and dragBoundsProperty can be provided, as they handle mapping of the drag point'
    );

    PressListener.call( this, options );

    // @private (stored options)
    this._allowTouchSnag = options.allowTouchSnag;
    this._applyOffset = options.applyOffset;
    this._trackAncestors = options.trackAncestors;
    this._translateNode = options.translateNode;
    this._transform = options.transform;
    this._locationProperty = options.locationProperty;
    this._mapLocation = options.mapLocation;
    this._offsetLocation = options.offsetLocation;
    this._dragBoundsProperty = ( options.dragBoundsProperty || new Property( null ) );
    this._start = options.start;
    this._end = options.end;

    // @public {Property.<boolean>} - Alias for isPressedProperty (as this name makes more sense for dragging)
    this.isUserControlledProperty = this.isPressedProperty;

    // @private {Vector2} - The point of the drag in the target's global coordinate frame. Updated with mutation.
    this._globalPoint = new Vector2( 0, 0 );

    // @private {Vector2} - The point of the drag in the target's local coordinate frame. Updated with mutation.
    this._localPoint = new Vector2( 0, 0 );

    // @private {Vector2} - Current drag point in the parent coordinate frame. Updated with mutation.
    this._parentPoint = new Vector2( 0, 0 );

    // @private {Vector2} - Current drag point in the model coordinate frame
    this._modelPoint = new Vector2( 0, 0 );

    // @private {TransformTracker|null} - Handles watching ancestor transforms for callbacks.
    this._transformTracker = null;

    // @private {Function} - Listener passed to the transform tracker
    this._transformTrackerListener = this.ancestorTransformed.bind( this );

    // @private {Pointer|null} - There are cases like https://github.com/phetsims/equality-explorer/issues/97 where if
    // a touchenter starts a drag that is IMMEDIATELY interrupted, the touchdown would start another drag. We record
    // interruptions here so that we can prevent future enter/down events from the same touch pointer from triggering
    // another startDrag.
    this._lastInterruptedTouchPointer = null;

    // @private {Emitter} - emitted on drag. Used for triggering phet-io events to the data stream, see https://github.com/phetsims/scenery/issues/842
    this._draggedAction = new Action( function( event ) {

      // This is done first, before the drag listener is called (from the prototype drag call)
      if ( !self._globalPoint.equals( self.pointer.point ) ) {
        self.reposition( self.pointer.point );
      }

      PressListener.prototype.drag.call( self, event );
    }, {
      phetioFeatured: options.phetioFeatured,
      tandem: options.tandem.createTandem( 'draggedAction' ),
      phetioHighFrequency: true,
      phetioDocumentation: 'Emits whenever a drag occurs with an EventIO argument.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: PhetioObject.EventType.USER,
      phetioType: DraggedActionIO
    } );
  }

  scenery.register( 'DragListener', DragListener );

  inherit( PressListener, DragListener, {
    /**
     * Attempts to start a drag with a press.
     * @public
     * @override
     *
     * NOTE: This is safe to call externally in order to attempt to start a press. dragListener.canPress( event ) can
     * be used to determine whether this will actually start a drag.
     *
     * @param {Event} event
     * @param {Node} [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
     *                              forwarded presses.
     * @param {function} [callback] - to be run at the end of the function, but only on success
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press: function( event, targetNode, callback ) {
      var self = this;
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      var success = PressListener.prototype.press.call( this, event, targetNode, function() {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener successful press' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        self.attachTransformTracker();

        // Set the local point
        self.parentToLocalPoint( self.globalToParentPoint( self._localPoint.set( self.pointer.point ) ) );

        self.reposition( self.pointer.point );

        // Notify after positioning and other changes
        self._start && self._start( event, self );

        callback && callback();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      } );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

      return success;
    },

    /**
     * DragListener should not be clicked for a11y, this prevents that listener from being called.
     * See https://github.com/phetsims/scenery/issues/903
     * @public
     * @returns {boolean}
     */
    canClick: function() {
      return false;
    },

    /**
     * Stops the drag.
     * @public
     * @override
     *
     * This can be called from the outside to stop the drag without the pointer having actually fired any 'up'
     * events. If the cancel/interrupt behavior is more preferable, call interrupt() on this listener instead.
     *
     * @param {function} [event] - scenery Event if there was one
     * @param {function} [callback] - called at the end of the release
     */
    release: function( event, callback ) {
      var self = this;

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener release' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      PressListener.prototype.release.call( this, event, function() {
        self.detachTransformTracker();

        // Notify after the rest of release is called in order to prevent it from triggering interrupt().
        self._end && self._end( self );

        callback && callback();
      } );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called when move events are fired on the attached pointer listener during a drag.
     * @protected
     * @override
     *
     * @param {Event} event
     */
    drag: function( event ) {
      // Ignore global moves that have zero length (Chrome might autofire, see
      // https://code.google.com/p/chromium/issues/detail?id=327114)
      if ( this._globalPoint.equals( this.pointer.point ) ) {
        return;
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener drag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this._draggedAction.execute( event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Attempts to start a touch snag, given a Scenery Event.
     * @public
     *
     * Should be safe to be called externally with an event.
     *
     * @param {Event} event
     */
    tryTouchSnag: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener tryTouchSnag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      if ( this._allowTouchSnag && !event.pointer.isAttached() ) {
        this.press( event );
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Returns a defensive copy of the local-coordinate-frame point of the drag.
     * @public
     *
     * @returns {Vector2}
     */
    getGlobalPoint: function() {
      return this._globalPoint.copy();
    },
    get globalPoint() { return this.getGlobalPoint(); },

    /**
     * Returns a defensive copy of the local-coordinate-frame point of the drag.
     * @public
     *
     * @returns {Vector2}
     */
    getLocalPoint: function() {
      return this._localPoint.copy();
    },
    get localPoint() { return this.getLocalPoint(); },

    /**
     * Returns a defensive copy of the parent-coordinate-frame point of the drag.
     * @public
     *
     * @returns {Vector2}
     */
    getParentPoint: function() {
      return this._parentPoint.copy();
    },
    get parentPoint() { return this.getParentPoint(); },

    /**
     * Returns a defensive copy of the model-coordinate-frame point of the drag.
     * @public
     *
     * @returns {Vector2}
     */
    getModelPoint: function() {
      return this._modelPoint.copy();
    },
    get modelPoint() { return this.getModelPoint(); },

    /**
     * Maps a point from the global coordinate frame to our drag target's parent coordinate frame.
     * @protected
     *
     * NOTE: This mutates the input vector (for performance)
     *
     * Should be overridden if a custom transformation is needed.
     *
     * @param {Vector2} globalPoint
     * @returns {Vector2}
     */
    globalToParentPoint: function( globalPoint ) {
      if ( assert ) {
        var referenceResult = this.pressedTrail.globalToParentPoint( globalPoint );
      }
      this.pressedTrail.getParentTransform().getInverse().multiplyVector2( globalPoint );
      assert && assert( globalPoint.equals( referenceResult ) );
      return globalPoint;
    },

    /**
     * Maps a point from the drag target's parent coordinate frame to its local coordinate frame.
     * @protected
     *
     * NOTE: This mutates the input vector (for performance)
     *
     * Should be overridden if a custom transformation is needed.
     *
     * @param {Vector2} parentPoint
     * @returns {Vector2}
     */
    parentToLocalPoint: function( parentPoint ) {
      if ( assert ) {
        var referenceResult = this.pressedTrail.lastNode().parentToLocalPoint( parentPoint );
      }
      this.pressedTrail.lastNode().getTransform().getInverse().multiplyVector2( parentPoint );
      assert && assert( parentPoint.equals( referenceResult ) );
      return parentPoint;
    },

    /**
     * Maps a point from the drag target's local coordinate frame to its parent coordinate frame.
     * @protected
     *
     * NOTE: This mutates the input vector (for performance)
     *
     * Should be overridden if a custom transformation is needed.
     *
     * @param {Vector2} localPoint
     * @returns {Vector2}
     */
    localToParentPoint: function( localPoint ) {
      if ( assert ) {
        var referenceResult = this.pressedTrail.lastNode().localToParentPoint( localPoint );
      }
      this.pressedTrail.lastNode().getMatrix().multiplyVector2( localPoint );
      assert && assert( localPoint.equals( referenceResult ) );
      return localPoint;
    },

    /**
     * Maps a point from the drag target's parent coordinate frame to the model coordinate frame.
     * @protected
     *
     * NOTE: This mutates the input vector (for performance)
     *
     * Should be overridden if a custom transformation is needed. Note that by default, unless a transform is provided,
     * the parent coordinate frame will be the same as the model coordinate frame.
     *
     * @param {Vector2} parentPoint
     * @returns {Vector2}
     */
    parentToModelPoint: function( parentPoint ) {
      if ( this._transform ) {
        this._transform.getInverse().multiplyVector2( parentPoint );
      }
      return parentPoint;
    },

    /**
     * Maps a point from the model coordinate frame to the drag target's parent coordinate frame.
     * @protected
     *
     * NOTE: This mutates the input vector (for performance)
     *
     * Should be overridden if a custom transformation is needed. Note that by default, unless a transform is provided,
     * the parent coordinate frame will be the same as the model coordinate frame.
     *
     * @param {Vector2} modelPoint
     * @returns {Vector2}
     */
    modelToParentPoint: function( modelPoint ) {
      if ( this._transform ) {
        this._transform.getMatrix().multiplyVector2( modelPoint );
      }
      return modelPoint;
    },

    /**
     * Apply a mapping from from the drag target's model location to an allowed model location.
     * @protected
     *
     * A common example is using dragBounds, where the location of the drag target is constrained to within a bounding
     * box. This is done by mapping points outside of the bounding box to the closest location inside the box. More
     * general mappings can be used.
     *
     * Should be overridden (or use mapLocation) if a custom transformation is needed.
     *
     * @param {Vector2} modelPoint
     * @returns {Vector2} - A point in the model coordinate frame
     */
    mapModelPoint: function( modelPoint ) {
      if ( this._mapLocation ) {
        return this._mapLocation( modelPoint );
      }
      else if ( this._dragBoundsProperty.value ) {
        return this._dragBoundsProperty.value.closestPointTo( modelPoint );
      }
      else {
        return modelPoint;
      }
    },

    /**
     * Mutates the parentPoint given to account for the initial pointer's offset from the drag target's origin.
     * @protected
     *
     * @param {Vector2} parentPoint
     */
    applyParentOffset: function( parentPoint ) {
      if ( this._offsetLocation ) {
        parentPoint.add( this._offsetLocation( parentPoint, this ) );
      }

      // Don't apply any offset if applyOffset is false
      if ( this._applyOffset ) {
        // Add the difference between our local origin (in the parent coordinate frame) and the local point (in the same
        // parent coordinate frame).
        parentPoint.subtract( this.localToParentPoint( scratchVector2A.set( this._localPoint ) ) );
        parentPoint.add( this.localToParentPoint( scratchVector2A.setXY( 0, 0 ) ) );
      }
    },

    /**
     * Triggers an update of the drag position, potentially changing location properties.
     * @public
     *
     * Should be called when something that changes the output locations of the drag occurs (most often, a drag event
     * itself).
     *
     * @param {Vector2} globalPoint
     */
    reposition: function( globalPoint ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener reposition' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this._globalPoint.set( globalPoint );

      // Update parentPoint mutably.
      this.applyParentOffset( this.globalToParentPoint( this._parentPoint.set( globalPoint ) ) );

      // Compute the modelPoint from the parentPoint
      this._modelPoint = this.mapModelPoint( this.parentToModelPoint( scratchVector2A.set( this._parentPoint ) ) );

      // Apply any mapping changes back to the parent point
      this.modelToParentPoint( this._parentPoint.set( this._modelPoint ) );

      if ( this._translateNode ) {
        this.pressedTrail.lastNode().translation = this._parentPoint;
      }

      if ( this._locationProperty ) {
        this._locationProperty.value = this._modelPoint.copy(); // Include an extra reference so that it will change.
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'touchenter' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly. See the press method instead.
     *
     * @param {Event} event
     */
    touchenter: function( event ) {
      this.tryTouchSnag( event );
    },

    /**
     * Called with 'touchmove' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly. See the press method instead.
     *
     * @param {Event} event
     */
    touchmove: function( event ) {
      this.tryTouchSnag( event );
    },

    /**
     * Called when an ancestor's transform has changed (when trackAncestors is true).
     * @private
     */
    ancestorTransformed: function() {
      // Reposition based on the current point.
      this.reposition( this.pointer.point );
    },

    /**
     * Attaches our transform tracker (begins listening to the ancestor transforms)
     * @private
     */
    attachTransformTracker: function() {
      if ( this._trackAncestors ) {
        this._transformTracker = new TransformTracker( this.pressedTrail.copy().removeDescendant() );
        this._transformTracker.addListener( this._transformTrackerListener );
      }
    },

    /**
     * Detaches our transform tracker (stops listening to the ancestor transforms)
     * @private
     */
    detachTransformTracker: function() {
      if ( this._transformTracker ) {
        this._transformTracker.removeListener( this._transformTrackerListener );
        this._transformTracker.dispose();
        this._transformTracker = null;
      }
    },

    /**
     * Sets the drag bounds of the listener.
     * @public
     *
     * @param {Bounds2} bounds
     */
    setDragBounds: function( bounds ) {
      assert && assert( bounds instanceof Bounds2 );

      this._dragBoundsProperty.value = bounds;
    },
    set dragBounds( value ) { this.setDragBounds( value ); },

    /**
     * Returns the drag bounds of the listener.
     * @public
     *
     * @returns {Bounds2}
     */
    getDragBounds: function() {
      return this._dragBoundsProperty.value;
    },
    get dragBounds() { return this.getDragBounds(); },

    /**
     * Interrupts the listener, releasing it (canceling behavior).
     * @public
     * @override
     *
     * This can be called manually, but can also be called through node.interruptSubtreeInput().
     */
    interrupt: function() {
      if ( this.pointer && this.pointer instanceof Touch ) {
        this._lastInterruptedTouchPointer = this.pointer;
      }

      PressListener.prototype.interrupt.call( this );
    },

    /**
     * Returns whether a press can be started with a particular event.
     * @public
     * @override
     *
     * @param {Event} event
     * @returns {boolean}
     */
    canPress: function( event ) {
      if ( event.pointer === this._lastInterruptedTouchPointer ) {
        return false;
      }

      return PressListener.prototype.canPress.call( this, event );
    },

    /**
     * Disposes the listener, releasing references. It should not be used after this.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener dispose' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this._draggedAction.dispose();

      this.detachTransformTracker();

      PressListener.prototype.dispose.call( this );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  }, {
    /**
     * Creates an input listener that forwards events to the specified input listener.
     * @public
     *
     * See https://github.com/phetsims/scenery/issues/639
     *
     * @param {function} down - function( {Event} ) - down function to be added to the input listener
     * @param {Object} [options]
     * @returns {Object} a scenery input listener
     */
    createForwardingListener: function( down, options ) {

      options = _.extend( {
        allowTouchSnag: false
      }, options );

      return {
        down: function( event ) {
          if ( event.canStartPress() ) {
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

  return DragListener;
} );

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
 *    transforming the Node if translateNode:true.
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
  var Vector2 = require( 'DOT/Vector2' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );
  var TransformTracker = require( 'SCENERY/util/TransformTracker' );

  /**
   * @constructor
   * @extends PressListener
   *
   * @param {Object} [options] - See the constructor body (below) and in PressListener for documented options.
   */
  function DragListener( options ) {
    options = _.extend( {
      // {boolean} - If true, unattached touches that move across our node will trigger a press(). This helps sometimes
      // for small draggable objects.
      allowTouchSnag: false,

      // {boolean} - If true, the initial offset of the pointer's location is taken into account, so that drags will
      // try to keep the pointer at the same local point of our dragged node.
      // TODO: how does this work if our node gets scaled/rotated during drag?
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

      // {Bounds2|null} - If provided, the model location will be constrained to be inside these bounds.
      // TODO: support mutability for this type of thing (or support Property.<Bounds2>)
      dragBounds: null,

      // {Function|null} - function( modelPoint: {Vector2} ) : {Vector2}. If provided, it will allow custom mapping
      // from the desired location (i.e. where the pointer is) to the actual possible location (i.e. where the dragged
      // object ends up). For example, using dragBounds is equivalent to passing:
      //   mapLocation: function( point ) { return dragBounds.closestPointTo( point ); }
      mapLocation: null,

      // {Function|null} - Called as start( event: {Event} ) when the drag is started. This is preferred over passing
      // press(), as the drag start hasn't been fully processed at that point.
      start: null,

      // {Function|null} - Called as end() when the drag is ended. This is preferred over passing release(), as the
      // drag start hasn't been fully processed at that point.
      end: null
    }, options );

    // TODO: type checks for options

    assert && assert(
      !options.mapLocation || !options.dragBounds,
      'mapLocation and dragBounds cannot both be provided, as they both handle mapping of the drag point'
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
    this._dragBounds = options.dragBounds;
    this._start = options.start;
    this._end = options.end;

    // TODO: scratch vectors?
    // @public {Vector2} - Initial point of the drag in the target's local coordinate frame
    this.initialLocalPoint = null;

    // @public {Vector2} - Current drag point in the parent coordinate frame
    this.parentPoint = new Vector2();

    // @public {Vector2} - Current drag point in the model coordinate frame
    this.modelPoint = new Vector2();

    // @private {TransformTracker|null} - Handles watching ancestor transforms for callbacks.
    this._transformTracker = null;

    // @private {Function} - Listener passed to the transform tracker
    this._transformTrackerListener = this.ancestorTransformed.bind( this );
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
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      var success = PressListener.prototype.press.call( this, event );

      if ( success ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener successful press' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.attachTransformTracker();

        // TODO: scratch vectors
        this.initialLocalPoint = this.parentToLocalPoint( this.globalToParentPoint( this.pointer.point ) );

        this.reposition( this.pointer.point );

        // Notify after positioning and other changes
        this._start && this._start( event );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

      return success;
    },

    /**
     * Stops the drag.
     * @public
     * @override
     *
     * This can be called from the outside to stop the drag without the pointer having actually fired any 'up'
     * events. If the cancel/interrupt behavior is more preferable, call interrupt() on this listener instead.
     */
    release: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener release' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      PressListener.prototype.release.call( this );

      this.detachTransformTracker();

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      // TODO: Is this a problem that we can't access things like this.pointer here?
      this._end && this._end();

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
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener drag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // This is done first, before the drag listener is called (from the prototype drag call)
      this.reposition( this.pointer.point );

      //TODO ignore global moves that have zero length (Chrome might autofire, see https://code.google.com/p/chromium/issues/detail?id=327114)
      //TODO: should this apply in PressListener's drag?
      PressListener.prototype.drag.call( this, event );

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
     * Maps a point from the global coordinate frame to our drag target's parent coordinate frame.
     * @protected
     *
     * Should be overridden if a custom transformation is needed.
     *
     * TODO: Reduce allocations. Will probably require augmenting Transform3 to use Matrix3 methods like multiplyVector2
     *
     * @param {Vector2} globalPoint
     * @returns {Vector2}
     */
    globalToParentPoint: function( globalPoint ) {
      return this.pressedTrail.globalToParentPoint( globalPoint );
    },

    /**
     * Maps a point from the drag target's parent coordinate frame to its local coordinate frame.
     * @protected
     *
     * Should be overridden if a custom transformation is needed.
     *
     * TODO: Reduce allocations. Will probably require augmenting Transform3 to use Matrix3 methods like multiplyVector2
     *
     * @param {Vector2} parentPoint
     * @returns {Vector2}
     */
    parentToLocalPoint: function( parentPoint ) {
      return this.pressedTrail.lastNode().parentToLocalPoint( parentPoint );
    },

    /**
     * Maps a point from the drag target's local coordinate frame to its parent coordinate frame.
     * @protected
     *
     * Should be overridden if a custom transformation is needed.
     *
     * TODO: Reduce allocations. Will probably require augmenting Transform3 to use Matrix3 methods like multiplyVector2
     *
     * @param {Vector2} localPoint
     * @returns {Vector2}
     */
    localToParentPoint: function( localPoint ) {
      return this.pressedTrail.lastNode().localToParentPoint( localPoint );
    },

    /**
     * Maps a point from the drag target's parent coordinate frame to the model coordinate frame.
     * @protected
     *
     * Should be overridden if a custom transformation is needed. Note that by default, unless a transform is provided,
     * the parent coordinate frame will be the same as the model coordinate frame.
     *
     * TODO: Reduce allocations. Will probably require augmenting Transform3 to use Matrix3 methods like multiplyVector2
     *
     * @param {Vector2} parentPoint
     * @returns {Vector2}
     */
    parentToModelPoint: function( parentPoint ) {
      if ( this._transform ) {
        return this._transform.inversePosition2( parentPoint );
      }
      else {
        return parentPoint;
      }
    },

    /**
     * Maps a point from the model coordinate frame to the drag target's parent coordinate frame.
     * @protected
     *
     * Should be overridden if a custom transformation is needed. Note that by default, unless a transform is provided,
     * the parent coordinate frame will be the same as the model coordinate frame.
     *
     * TODO: Reduce allocations. Will probably require augmenting Transform3 to use Matrix3 methods like multiplyVector2
     *
     * @param {Vector2} modelPoint
     * @returns {Vector2}
     */
    modelToParentPoint: function( modelPoint ) {
      if ( this._transform ) {
        return this._transform.transformPosition2( modelPoint );
      }
      else {
        return modelPoint;
      }
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
     * TODO: consider mutating the point if possible? Can closestPointTo do that?
     *
     * @param {Vector2} modelPoint
     * @returns {Vector2} - A point in the model coordinate frame
     */
    mapModelPoint: function( modelPoint ) {
      if ( this._mapLocation ) {
        return this._mapLocation( modelPoint );
      }
      else if ( this._dragBounds ) {
        return this._dragBounds.closestPointTo( modelPoint );
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
      // Don't apply any offset if applyOffset is false
      if ( this._applyOffset ) {
        // TODO: more scratch vector handling? Will need to augment Transform3 to use things like multiplyVector2.
        var parentLocalPoint = this.localToParentPoint( this.initialLocalPoint );
        var parentOriginPoint = this.localToParentPoint( new Vector2() ); // usually node.translation
        var parentOffset = parentOriginPoint.minus( parentLocalPoint );

        parentPoint.add( parentOffset );
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

      // TODO: scratch vectors
      this.parentPoint = this.globalToParentPoint( globalPoint );
      this.applyParentOffset( this.parentPoint );
      this.modelPoint = this.mapModelPoint( this.parentToModelPoint( this.parentPoint ) );
      this.parentPoint = this.modelToParentPoint( this.modelPoint ); // apply any mapping changes

      if ( this._translateNode ) {
        this.pressedTrail.lastNode().translation = this.parentPoint;
      }

      if ( this._locationProperty ) {
        this._locationProperty.value = this.modelPoint;
      }

      // TODO: consider other options to handle here. Should deltas (global/parent/local/model) be provided?

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
     * Disposes the listener, releasing references. It should not be used after this.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener dispose' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.detachTransformTracker();

      PressListener.prototype.dispose.call( this );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  } );

  return DragListener;
} );

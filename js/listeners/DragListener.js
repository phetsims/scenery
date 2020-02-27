// Copyright 2017-2020, University of Colorado Boulder

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
 * 1. When a drag is started (with press), record the pointer's position in the local coordinate frame. This is visually
 *    where the pointer is over the drag target, and typically most drags will want to move the dragged element so that
 *    the pointer continues to be over this point.
 * 2. When the pointer is moved, compute the new parent translation to keep the pointer on the same place on the
 *    dragged element.
 * 3. (optionally) map that to a model position, and (optionally) move that model position to satisfy any constraints of
 *    where the element can be dragged (recomputing the parent/model translation as needed)
 * 4. Apply the required translation (with a provided drag callback, using the positionProperty, or directly
 *    transforming the Node if translateNode:true).
 *
 * For example usage, see scenery/examples/input.html
 *
 * For most PhET model-view usage, it's recommended to include a model position Property as the `positionProperty`
 * option, along with the `transform` option specifying the MVT. By default, this will then assume that the Node with
 * the listener is positioned in the "view" coordinate frame, and will properly handle offsets and transformations.
 * It is assumed that when the model `positionProperty` changes, that the position of the Node would also change.
 * If it's another Node being transformed, please use the `targetNode` option to specify which Node is being
 * transformed. If something more complicated than a Node being transformed is going on (like positioning multiple
 * items, positioning based on the center, changing something in CanvasNode), it's recommended to pass the
 * `useParentOffset` option (so that the DragListener will NOT try to compute offsets based on the Node's position), or
 * to use `applyOffset:false` (effectively having drags reposition the Node so that the origin is at the pointer).
 *
 * The typical PhET usage would look like:
 *
 *   new DragListener( {
 *     positionProperty: someObject.positionProperty,
 *     transform: modelViewTransform
 *   } )
 *
 * Additionally, for PhET usage it's also fine NOT to hook into a `positionProperty`. Typically using start/end/drag,
 * and values can be read out (like `modelPoint`, `localPoint`, `parentPoint`, `modelDelta`) from the listener to do
 * operations. For instance, if deltas and model positions are the only thing desired:
 *
 *   new DragListener( {
 *     drag: ( event, listener ) => {
 *       doSomethingWith( listener.modelDelta, listener.modelPoint );
 *     }
 *   } )
 *
 * It's completely fine to use one DragListener with multiple objects, however this isn't done as much since specifying
 * positionProperty only works with ONE model position Property (so if things are backed by the same Property it would
 * be fine). Doing things based on modelPoint/modelDelta/etc. should be completely fine using one listener with
 * multiple nodes. The typical pattern IS creating one DragListener per draggable view Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Action from '../../../axon/js/Action.js';
import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import inherit from '../../../phet-core/js/inherit.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import SceneryEventIO from '../input/SceneryEventIO.js';
import Touch from '../input/Touch.js';
import scenery from '../scenery.js';
import TransformTracker from '../util/TransformTracker.js';
import PressListener from './PressListener.js';

// Scratch vectors used to prevent allocations
const scratchVector2A = new Vector2( 0, 0 );

/**
 * @constructor
 * @extends PressListener
 *
 * @param {Object} [options] - See the constructor body (below) and in PressListener for documented options.
 */
function DragListener( options ) {
  options = merge( {

    // {Property.<Vector2>|null} - If provided, it will be synchronized with the drag position in the model coordinate
    // frame (applying any provided transforms as needed). Typically, DURING a drag this Property should not be
    // modified externally (as the next drag event will probably undo the change), but it's completely fine to modify
    // this Property at any other time.
    positionProperty: null,

    // {Function|null} - Called as start( event: {SceneryEvent}, listener: {DragListener} ) when the drag is started.
    // This is preferred over passing press(), as the drag start hasn't been fully processed at that point.
    start: null,

    // {Function|null} - Called as end( listener: {DragListener} ) when the drag is ended. This is preferred over
    // passing release(), as the drag start hasn't been fully processed at that point.
    // NOTE: This will also be called if the drag is ended due to being interrupted or canceled.
    end: null,

    // {Transform3|null} - If provided, this will be the conversion between the parent (view) and model coordinate
    // frames. Usually most useful when paired with the positionProperty.
    transform: null,

    // {Property.<Bounds2|null>} - If provided, the model position will be constrained to be inside these bounds.
    dragBoundsProperty: null,

    // {boolean} - If true, unattached touches that move across our node will trigger a press(). This helps sometimes
    // for small draggable objects.
    allowTouchSnag: true,

    // {boolean} - If true, the initial offset of the pointer's position is taken into account, so that drags will
    // try to keep the pointer at the same local point of our dragged node.
    // NOTE: The default behavior is to use the given Node (either the targetNode or the node with the listener on it)
    // and use its transform to compute the "local point" (assuming that the node's local origin is what is
    // transformed around). This is ideal for most situations, but it's also possible to use a parent-coordinate
    // based approach for offsets (see useParentOffset)
    applyOffset: true,

    // {boolean} - If set to true, then any offsets applied will be handled in the parent coordinate space using the
    // positionProperty as the "ground truth", instead of looking at the Node's actual position and transform. This
    // is useful if the position/transform cannot be applied directly to a single Node (e.g. positioning multiple
    // independent nodes, or centering things instead of transforming based on the origin of the Node).
    //
    // NOTE: Use this option most likely if converting from MoveableDragHandler, because it transformed based in
    // the parent's coordinate frame. See https://github.com/phetsims/scenery/issues/1014
    useParentOffset: false,

    // {boolean} - If true, ancestor transforms will be watched. If they change, it will trigger a repositioning,
    // which will usually adjust the position/transform to maintain position.
    trackAncestors: false,

    // {boolean} - If true, the effective currentTarget will be translated when the drag position changes.
    translateNode: false,

    // {Function|null} - function( modelPoint: {Vector2} ) : {Vector2}. If provided, it will allow custom mapping
    // from the desired position (i.e. where the pointer is) to the actual possible position (i.e. where the dragged
    // object ends up). For example, using dragBoundsProperty is equivalent to passing:
    //   mapPosition: function( point ) { return dragBoundsProperty.value.closestPointTo( point ); }
    mapPosition: null,

    // {Function|null} - function( viewPoint: {Vector2}, listener: {DragListener} ) : {Vector2}. If provided, its
    // result will be added to the parentPoint before computation continues, to allow the ability to "offset" where
    // the pointer position seems to be. Useful for touch, where things shouldn't be under the pointer directly.
    offsetPosition: null,

    // {Tandem} - For instrumenting
    tandem: Tandem.REQUIRED,

    // to support properly passing this to children, see https://github.com/phetsims/tandem/issues/60
    phetioReadOnly: PhetioObject.DEFAULT_OPTIONS.phetioReadOnly,
    phetioFeatured: PhetioObject.DEFAULT_OPTIONS.phetioFeatured
  }, options );

  assert && assert( typeof options.allowTouchSnag === 'boolean', 'allowTouchSnag should be a boolean' );
  assert && assert( typeof options.applyOffset === 'boolean', 'applyOffset should be a boolean' );
  assert && assert( typeof options.trackAncestors === 'boolean', 'trackAncestors should be a boolean' );
  assert && assert( typeof options.translateNode === 'boolean', 'translateNode should be a boolean' );
  assert && assert( options.transform === null || options.transform instanceof Transform3, 'transform, if provided, should be a Transform3' );
  assert && assert( options.positionProperty === null || options.positionProperty instanceof Property, 'positionProperty, if provided, should be a Property' );
  assert && assert( !options.dragBounds, 'options.dragBounds was removed in favor of options.dragBoundsProperty' );
  assert && assert( options.dragBoundsProperty === null || options.dragBoundsProperty instanceof Property, 'dragBoundsProperty, if provided, should be a Property' );
  assert && assert( options.mapPosition === null || typeof options.mapPosition === 'function', 'mapPosition, if provided, should be a function' );
  assert && assert( options.offsetPosition === null || typeof options.offsetPosition === 'function', 'offsetPosition, if provided, should be a function' );
  assert && assert( options.start === null || typeof options.start === 'function', 'start, if provided, should be a function' );
  assert && assert( options.end === null || typeof options.end === 'function', 'end, if provided, should be a function' );
  assert && assert( options.tandem instanceof Tandem, 'The provided tandem should be a Tandem' );
  assert && assert( !options.useParentOffset || options.positionProperty, 'If useParentOffset is set, a positionProperty is required' );

  assert && assert(
    !( options.mapPosition && options.dragBoundsProperty ),
    'Only one of mapPosition and dragBoundsProperty can be provided, as they handle mapping of the drag point'
  );

  PressListener.call( this, options );

  // @private (stored options)
  this._allowTouchSnag = options.allowTouchSnag;
  this._applyOffset = options.applyOffset;
  this._useParentOffset = options.useParentOffset;
  this._trackAncestors = options.trackAncestors;
  this._translateNode = options.translateNode;
  this._transform = options.transform;
  this._positionProperty = options.positionProperty;
  this._mapPosition = options.mapPosition;
  this._offsetPosition = options.offsetPosition;
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

  // @private {Vector2} - Stores the model delta computed during every repositioning
  this._modelDelta = new Vector2( 0, 0 );

  // @private {Vector2} - If useParentOffset is true, this will be set to the parent-coordinate offset at the start
  // of a drag, and the "offset" will be handled by applying this offset compared to where the pointer is.
  this._parentOffset = new Vector2( 0, 0 );

  // @private {TransformTracker|null} - Handles watching ancestor transforms for callbacks.
  this._transformTracker = null;

  // @private {Function} - Listener passed to the transform tracker
  this._transformTrackerListener = this.ancestorTransformed.bind( this );

  // @private {Pointer|null} - There are cases like https://github.com/phetsims/equality-explorer/issues/97 where if
  // a touchenter starts a drag that is IMMEDIATELY interrupted, the touchdown would start another drag. We record
  // interruptions here so that we can prevent future enter/down events from the same touch pointer from triggering
  // another startDrag.
  this._lastInterruptedTouchPointer = null;

  // @private {Action} - emitted on drag. Used for triggering phet-io events to the data stream, see https://github.com/phetsims/scenery/issues/842
  this._dragAction = new Action( event => {

    // This is done first, before the drag listener is called (from the prototype drag call)
    if ( !this._globalPoint.equals( this.pointer.point ) ) {
      this.reposition( this.pointer.point );
    }

    PressListener.prototype.drag.call( this, event );
  }, {
    parameters: [ { name: 'event', phetioType: SceneryEventIO } ],
    phetioFeatured: options.phetioFeatured,
    tandem: options.tandem.createTandem( 'dragAction' ),
    phetioHighFrequency: true,
    phetioDocumentation: 'Emits whenever a drag occurs with an SceneryEventIO argument.',
    phetioReadOnly: options.phetioReadOnly,
    phetioEventType: EventType.USER
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
   * @param {SceneryEvent} event
   * @param {Node} [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
   *                              forwarded presses.
   * @param {function} [callback] - to be run at the end of the function, but only on success
   * @returns {boolean} success - Returns whether the press was actually started
   */
  press( event, targetNode, callback ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener press' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    const success = PressListener.prototype.press.call( this, event, targetNode, () => {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener successful press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.attachTransformTracker();

      // Compute the parent point corresponding to the pointer's position
      const parentPoint = this.globalToParentPoint( this._localPoint.set( this.pointer.point ) );

      if ( this._useParentOffset ) {
        this.modelToParentPoint( this._parentOffset.set( this._positionProperty.value ) ).subtract( parentPoint );
      }

      // Set the local point
      this.parentToLocalPoint( parentPoint );

      this.reposition( this.pointer.point );

      // Notify after positioning and other changes
      this._start && this._start( event, this );

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
  canClick() {
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
   * @param {SceneryEvent} [event] - scenery event if there was one
   * @param {function} [callback] - called at the end of the release
   */
  release( event, callback ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener release' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    PressListener.prototype.release.call( this, event, () => {
      this.detachTransformTracker();

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      this._end && this._end( this );

      callback && callback();
    } );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  },

  /**
   * Called when move events are fired on the attached pointer listener during a drag.
   * @protected
   * @override
   *
   * @param {SceneryEvent} event
   */
  drag( event ) {
    // Ignore global moves that have zero length (Chrome might autofire, see
    // https://code.google.com/p/chromium/issues/detail?id=327114)
    if ( this._globalPoint.equals( this.pointer.point ) ) {
      return;
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener drag' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._dragAction.execute( event );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  },

  /**
   * Attempts to start a touch snag, given a SceneryEvent.
   * @public
   *
   * Should be safe to be called externally with an event.
   *
   * @param {SceneryEvent} event
   */
  tryTouchSnag( event ) {
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
  getGlobalPoint() {
    return this._globalPoint.copy();
  },
  get globalPoint() { return this.getGlobalPoint(); },

  /**
   * Returns a defensive copy of the local-coordinate-frame point of the drag.
   * @public
   *
   * @returns {Vector2}
   */
  getLocalPoint() {
    return this._localPoint.copy();
  },
  get localPoint() { return this.getLocalPoint(); },

  /**
   * Returns a defensive copy of the parent-coordinate-frame point of the drag.
   * @public
   *
   * @returns {Vector2}
   */
  getParentPoint() {
    return this._parentPoint.copy();
  },
  get parentPoint() { return this.getParentPoint(); },

  /**
   * Returns a defensive copy of the model-coordinate-frame point of the drag.
   * @public
   *
   * @returns {Vector2}
   */
  getModelPoint() {
    return this._modelPoint.copy();
  },
  get modelPoint() { return this.getModelPoint(); },

  /**
   * Returns a defensive copy of the model-coordinate-frame delta.
   * @public
   *
   * @returns {Vector2}
   */
  getModelDelta() {
    return this._modelDelta.copy();
  },
  get modelDelta() { return this.getModelDelta(); },

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
  globalToParentPoint( globalPoint ) {
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
  parentToLocalPoint( parentPoint ) {
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
  localToParentPoint( localPoint ) {
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
  parentToModelPoint( parentPoint ) {
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
  modelToParentPoint( modelPoint ) {
    if ( this._transform ) {
      this._transform.getMatrix().multiplyVector2( modelPoint );
    }
    return modelPoint;
  },

  /**
   * Apply a mapping from from the drag target's model position to an allowed model position.
   * @protected
   *
   * A common example is using dragBounds, where the position of the drag target is constrained to within a bounding
   * box. This is done by mapping points outside of the bounding box to the closest position inside the box. More
   * general mappings can be used.
   *
   * Should be overridden (or use mapPosition) if a custom transformation is needed.
   *
   * @param {Vector2} modelPoint
   * @returns {Vector2} - A point in the model coordinate frame
   */
  mapModelPoint( modelPoint ) {
    if ( this._mapPosition ) {
      return this._mapPosition( modelPoint );
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
  applyParentOffset( parentPoint ) {
    if ( this._offsetPosition ) {
      parentPoint.add( this._offsetPosition( parentPoint, this ) );
    }

    // Don't apply any offset if applyOffset is false
    if ( this._applyOffset ) {
      if ( this._useParentOffset ) {
        parentPoint.add( this._parentOffset );
      }
      else {
        // Add the difference between our local origin (in the parent coordinate frame) and the local point (in the same
        // parent coordinate frame).
        parentPoint.subtract( this.localToParentPoint( scratchVector2A.set( this._localPoint ) ) );
        parentPoint.add( this.localToParentPoint( scratchVector2A.setXY( 0, 0 ) ) );
      }
    }
  },

  /**
   * Triggers an update of the drag position, potentially changing position properties.
   * @public
   *
   * Should be called when something that changes the output positions of the drag occurs (most often, a drag event
   * itself).
   *
   * @param {Vector2} globalPoint
   */
  reposition( globalPoint ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener reposition' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._globalPoint.set( globalPoint );

    // Update parentPoint mutably.
    this.applyParentOffset( this.globalToParentPoint( this._parentPoint.set( globalPoint ) ) );

    // To compute the delta (new - old), we first mutate it to (-old)
    this._modelDelta.set( this._modelPoint ).negate();

    // Compute the modelPoint from the parentPoint
    this._modelPoint = this.mapModelPoint( this.parentToModelPoint( scratchVector2A.set( this._parentPoint ) ) );

    // Complete the delta computation
    this._modelDelta.add( this._modelPoint );

    // Apply any mapping changes back to the parent point
    this.modelToParentPoint( this._parentPoint.set( this._modelPoint ) );

    if ( this._translateNode ) {
      this.pressedTrail.lastNode().translation = this._parentPoint;
    }

    if ( this._positionProperty ) {
      this._positionProperty.value = this._modelPoint.copy(); // Include an extra reference so that it will change.
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  },

  /**
   * Called with 'touchenter' events (part of the listener API).
   * @public (scenery-internal)
   *
   * NOTE: Do not call directly. See the press method instead.
   *
   * @param {SceneryEvent} event
   */
  touchenter( event ) {
    this.tryTouchSnag( event );
  },

  /**
   * Called with 'touchmove' events (part of the listener API).
   * @public (scenery-internal)
   *
   * NOTE: Do not call directly. See the press method instead.
   *
   * @param {SceneryEvent} event
   */
  touchmove( event ) {
    this.tryTouchSnag( event );
  },

  /**
   * Called when an ancestor's transform has changed (when trackAncestors is true).
   * @private
   */
  ancestorTransformed() {
    // Reposition based on the current point.
    this.reposition( this.pointer.point );
  },

  /**
   * Attaches our transform tracker (begins listening to the ancestor transforms)
   * @private
   */
  attachTransformTracker() {
    if ( this._trackAncestors ) {
      this._transformTracker = new TransformTracker( this.pressedTrail.copy().removeDescendant() );
      this._transformTracker.addListener( this._transformTrackerListener );
    }
  },

  /**
   * Detaches our transform tracker (stops listening to the ancestor transforms)
   * @private
   */
  detachTransformTracker() {
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
  setDragBounds( bounds ) {
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
  getDragBounds() {
    return this._dragBoundsProperty.value;
  },
  get dragBounds() { return this.getDragBounds(); },

  /**
   * Sets the drag transform of the listener.
   * @public
   *
   * @param {Bounds2} transform
   */
  setTransform( transform ) {
    assert && assert( transform instanceof Transform3 );

    this._transform = transform;
  },
  set transform( transform ) { this.setTransform( transform ); },

  /**
   * Returns the transform of the listener.
   * @public
   *
   * @returns {Transform3}
   */
  getTransform() {
    return this._transform;
  },
  get transform() { return this.getTransform(); },

  /**
   * Interrupts the listener, releasing it (canceling behavior).
   * @public
   * @override
   *
   * This effectively releases/ends the press, and sets the `interrupted` flag to true while firing these events
   * so that code can determine whether a release/end happened naturally, or was canceled in some way.
   *
   * This can be called manually, but can also be called through node.interruptSubtreeInput().
   */
  interrupt() {
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
   * @param {SceneryEvent} event
   * @returns {boolean}
   */
  canPress( event ) {
    if ( event.pointer === this._lastInterruptedTouchPointer ) {
      return false;
    }

    return PressListener.prototype.canPress.call( this, event );
  },

  /**
   * Disposes the listener, releasing references. It should not be used after this.
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener dispose' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._dragAction.dispose();

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
   * @param {function} down - function( {SceneryEvent} ) - down function to be added to the input listener
   * @param {Object} [options]
   * @returns {Object} a scenery input listener
   */
  createForwardingListener( down, options ) {

    options = merge( {
      allowTouchSnag: true // see https://github.com/phetsims/scenery/issues/999
    }, options );

    return {
      down( event ) {
        if ( event.canStartPress() ) {
          down( event );
        }
      },
      touchenter( event ) {
        options.allowTouchSnag && this.down( event );
      },
      touchmove( event ) {
        options.allowTouchSnag && this.down( event );
      }
    };
  }
} );

export default DragListener;
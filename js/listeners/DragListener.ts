// Copyright 2017-2023, University of Colorado Boulder

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

import PhetioAction from '../../../tandem/js/PhetioAction.js';
import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import optionize from '../../../phet-core/js/optionize.js';
import RequiredOption from '../../../phet-core/js/types/RequiredOption.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { Node, Pointer, PressedPressListener, PressListener, PressListenerCallback, PressListenerEvent, PressListenerNullableCallback, PressListenerOptions, scenery, SceneryEvent, TInputListener, TransformTracker } from '../imports.js';
import Property from '../../../axon/js/Property.js';

// Scratch vectors used to prevent allocations
const scratchVector2A = new Vector2( 0, 0 );

type MapPosition = ( point: Vector2 ) => Vector2;
type OffsetPosition<Listener extends DragListener> = ( point: Vector2, listener: Listener ) => Vector2;

type SelfOptions<Listener extends DragListener> = {
  // If provided, it will be synchronized with the drag position in the model coordinate
  // frame (applying any provided transforms as needed). Typically, DURING a drag this Property should not be
  // modified externally (as the next drag event will probably undo the change), but it's completely fine to modify
  // this Property at any other time.
  positionProperty?: TProperty<Vector2> | null;

  // Called as start( event: {SceneryEvent}, listener: {DragListener} ) when the drag is started.
  // This is preferred over passing press(), as the drag start hasn't been fully processed at that point.
  start?: PressListenerCallback<Listener> | null;

  // Called as end( listener: {DragListener} ) when the drag is ended. This is preferred over
  // passing release(), as the drag start hasn't been fully processed at that point.
  // NOTE: This will also be called if the drag is ended due to being interrupted or canceled.
  end?: PressListenerNullableCallback<Listener> | null;

  // If provided, this will be the conversion between the parent (view) and model coordinate
  // frames. Usually most useful when paired with the positionProperty.
  transform?: Transform3 | TReadOnlyProperty<Transform3> | null;

  // If provided, the model position will be constrained to be inside these bounds.
  dragBoundsProperty?: TReadOnlyProperty<Bounds2 | null> | null;

  // If true, unattached touches that move across our node will trigger a press(). This helps sometimes
  // for small draggable objects.
  allowTouchSnag?: boolean;

  // If true, the initial offset of the pointer's position is taken into account, so that drags will
  // try to keep the pointer at the same local point of our dragged node.
  // NOTE: The default behavior is to use the given Node (either the targetNode or the node with the listener on it)
  // and use its transform to compute the "local point" (assuming that the node's local origin is what is
  // transformed around). This is ideal for most situations, but it's also possible to use a parent-coordinate
  // based approach for offsets (see useParentOffset)
  applyOffset?: boolean;

  // If set to true, then any offsets applied will be handled in the parent coordinate space using the
  // positionProperty as the "ground truth", instead of looking at the Node's actual position and transform. This
  // is useful if the position/transform cannot be applied directly to a single Node (e.g. positioning multiple
  // independent nodes, or centering things instead of transforming based on the origin of the Node).
  //
  // NOTE: Use this option most likely if converting from MovableDragHandler, because it transformed based in
  // the parent's coordinate frame. See https://github.com/phetsims/scenery/issues/1014
  //
  // NOTE: This also requires providing a positionProperty
  useParentOffset?: boolean;

  // If true, ancestor transforms will be watched. If they change, it will trigger a repositioning;
  // which will usually adjust the position/transform to maintain position.
  trackAncestors?: boolean;

  // If true, the effective currentTarget will be translated when the drag position changes.
  translateNode?: boolean;

  // If provided, it will allow custom mapping
  // from the desired position (i.e. where the pointer is) to the actual possible position (i.e. where the dragged
  // object ends up). For example, using dragBoundsProperty is equivalent to passing:
  //   mapPosition: function( point ) { return dragBoundsProperty.value.closestPointTo( point ); }
  mapPosition?: MapPosition | null;

  // If provided, its result will be added to the parentPoint before computation continues, to allow the ability to
  // "offset" where the pointer position seems to be. Useful for touch, where things shouldn't be under the pointer
  // directly.
  offsetPosition?: OffsetPosition<Listener> | null;

  // pdom
  // Whether to allow `click` events to trigger behavior in the supertype PressListener.
  // Generally DragListener should not respond to click events, but there are some exceptions where drag
  // functionality is nice but a click should still activate the component. See
  // https://github.com/phetsims/sun/issues/696
  canClick?: boolean;
};
export type DragListenerOptions<Listener extends DragListener> = SelfOptions<Listener> & PressListenerOptions<Listener>;
type CreateForwardingListenerOptions = {
  allowTouchSnag?: boolean;
};

export type PressedDragListener = DragListener & PressedPressListener;
const isPressedListener = ( listener: DragListener ): listener is PressedDragListener => listener.isPressed;

export default class DragListener extends PressListener implements TInputListener {

  // Alias for isPressedProperty (as this name makes more sense for dragging)
  public isUserControlledProperty: TProperty<boolean>;

  private _allowTouchSnag: RequiredOption<SelfOptions<DragListener>, 'allowTouchSnag'>;
  private _applyOffset: RequiredOption<SelfOptions<DragListener>, 'applyOffset'>;
  private _useParentOffset: RequiredOption<SelfOptions<DragListener>, 'useParentOffset'>;
  private _trackAncestors: RequiredOption<SelfOptions<DragListener>, 'trackAncestors'>;
  private _translateNode: RequiredOption<SelfOptions<DragListener>, 'translateNode'>;
  private _transform: RequiredOption<SelfOptions<DragListener>, 'transform'>;
  private _positionProperty: RequiredOption<SelfOptions<DragListener>, 'positionProperty'>;
  private _mapPosition: RequiredOption<SelfOptions<DragListener>, 'mapPosition'>;
  private _offsetPosition: RequiredOption<SelfOptions<PressedDragListener>, 'offsetPosition'>;
  private _dragBoundsProperty: NonNullable<RequiredOption<SelfOptions<DragListener>, 'dragBoundsProperty'>>;
  private _start: RequiredOption<SelfOptions<PressedDragListener>, 'start'>;
  private _end: RequiredOption<SelfOptions<PressedDragListener>, 'end'>;
  private _canClick: RequiredOption<SelfOptions<DragListener>, 'canClick'>;

  // The point of the drag in the target's global coordinate frame. Updated with mutation.
  private _globalPoint: Vector2;

  // The point of the drag in the target's local coordinate frame. Updated with mutation.
  private _localPoint: Vector2;

  // Current drag point in the parent coordinate frame. Updated with mutation.
  private _parentPoint: Vector2;

  // Current drag point in the model coordinate frame
  private _modelPoint: Vector2;

  // Stores the model delta computed during every repositioning
  private _modelDelta: Vector2;

  // If useParentOffset is true, this will be set to the parent-coordinate offset at the start
  // of a drag, and the "offset" will be handled by applying this offset compared to where the pointer is.
  private _parentOffset: Vector2;

  // Handles watching ancestor transforms for callbacks.
  private _transformTracker: TransformTracker | null;

  // Listener passed to the transform tracker
  private _transformTrackerListener: () => void;

  // There are cases like https://github.com/phetsims/equality-explorer/issues/97 where if
  // a touchenter starts a drag that is IMMEDIATELY interrupted, the touchdown would start another drag. We record
  // interruptions here so that we can prevent future enter/down events from the same touch pointer from triggering
  // another startDrag.
  private _lastInterruptedTouchLikePointer: Pointer | null;

  // Emitted on drag. Used for triggering phet-io events to the data stream, see https://github.com/phetsims/scenery/issues/842
  private _dragAction: PhetioAction<[ PressListenerEvent ]>;


  public constructor( providedOptions?: DragListenerOptions<PressedDragListener> ) {
    const options = optionize<DragListenerOptions<PressedDragListener>, SelfOptions<PressedDragListener>, PressListenerOptions<PressedDragListener>>()( {
      positionProperty: null,
      start: null,
      end: null,
      transform: null,
      dragBoundsProperty: null,
      allowTouchSnag: true,
      applyOffset: true,
      useParentOffset: false,
      trackAncestors: false,
      translateNode: false,
      mapPosition: null,
      offsetPosition: null,
      canClick: false,

      tandem: Tandem.REQUIRED,

      // Though DragListener is not instrumented, declare these here to support properly passing this to children, see https://github.com/phetsims/tandem/issues/60.
      // DragListener by default doesn't allow PhET-iO to trigger drag Action events
      phetioReadOnly: true,
      phetioFeatured: PhetioObject.DEFAULT_OPTIONS.phetioFeatured
    }, providedOptions );

    assert && assert( !( options as unknown as { dragBounds: Bounds2 } ).dragBounds, 'options.dragBounds was removed in favor of options.dragBoundsProperty' );
    assert && assert( !options.useParentOffset || options.positionProperty, 'If useParentOffset is set, a positionProperty is required' );

    assert && assert(
      !( options.mapPosition && options.dragBoundsProperty ),
      'Only one of mapPosition and dragBoundsProperty can be provided, as they handle mapping of the drag point'
    );

    // @ts-expect-error TODO: See https://github.com/phetsims/phet-core/issues/128
    super( options );

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
    this._canClick = options.canClick;
    this.isUserControlledProperty = this.isPressedProperty;
    this._globalPoint = new Vector2( 0, 0 );
    this._localPoint = new Vector2( 0, 0 );
    this._parentPoint = new Vector2( 0, 0 );
    this._modelPoint = new Vector2( 0, 0 );
    this._modelDelta = new Vector2( 0, 0 );
    this._parentOffset = new Vector2( 0, 0 );
    this._transformTracker = null;
    this._transformTrackerListener = this.ancestorTransformed.bind( this );
    this._lastInterruptedTouchLikePointer = null;

    this._dragAction = new PhetioAction( event => {
      assert && assert( isPressedListener( this ) );
      const pressedListener = this as PressedDragListener;

      const point = pressedListener.pointer.point;

      if ( point ) {
        // This is done first, before the drag listener is called (from the prototype drag call)
        if ( !this._globalPoint.equals( point ) ) {
          this.reposition( point );
        }
      }

      PressListener.prototype.drag.call( this, event );
    }, {
      parameters: [ { name: 'event', phetioType: SceneryEvent.SceneryEventIO } ],
      phetioFeatured: options.phetioFeatured,
      tandem: options.tandem.createTandem( 'dragAction' ),
      phetioHighFrequency: true,
      phetioDocumentation: 'Emits whenever a drag occurs with an SceneryEventIO argument.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );
  }

  /**
   * Attempts to start a drag with a press.
   *
   * NOTE: This is safe to call externally in order to attempt to start a press. dragListener.canPress( event ) can
   * be used to determine whether this will actually start a drag.
   *
   * @param event
   * @param [targetNode] - If provided, will take the place of the targetNode for this call. Useful for forwarded presses.
   * @param [callback] - to be run at the end of the function, but only on success
   * @returns success - Returns whether the press was actually started
   */
  public override press( event: PressListenerEvent, targetNode?: Node, callback?: () => void ): boolean {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener press' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    const success = super.press( event, targetNode, () => {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener successful press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( isPressedListener( this ) );
      const pressedListener = this as PressedDragListener;

      // signify that this listener is reserved for dragging so that other listeners can change
      // their behavior during scenery event dispatch
      pressedListener.pointer.reserveForDrag();

      this.attachTransformTracker();

      assert && assert( pressedListener.pointer.point !== null );
      const point = pressedListener.pointer.point;

      // Compute the parent point corresponding to the pointer's position
      const parentPoint = this.globalToParentPoint( this._localPoint.set( point ) );

      if ( this._useParentOffset ) {
        this.modelToParentPoint( this._parentOffset.set( this._positionProperty!.value ) ).subtract( parentPoint );
      }

      // Set the local point
      this.parentToLocalPoint( parentPoint );

      this.reposition( point );

      // Notify after positioning and other changes
      this._start && this._start( event, pressedListener );

      callback && callback();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    } );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

    return success;
  }

  /**
   * Stops the drag.
   *
   * This can be called from the outside to stop the drag without the pointer having actually fired any 'up'
   * events. If the cancel/interrupt behavior is more preferable, call interrupt() on this listener instead.
   *
   * @param [event] - scenery event if there was one
   * @param [callback] - called at the end of the release
   */
  public override release( event?: PressListenerEvent, callback?: () => void ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener release' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    super.release( event, () => {
      this.detachTransformTracker();

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      this._end && this._end( event || null, this as PressedDragListener );

      callback && callback();
    } );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Components using DragListener should generally not be activated with a click. A single click from alternative
   * input would pick up the component then immediately release it. But occasionally that is desirable and can be
   * controlled with the canClick option.
   */
  public override canClick(): boolean {
    return super.canClick() && this._canClick;
  }

  /**
   * Activate the DragListener with a click activation. Usually, DragListener will NOT be activated with a click
   * and canClick will return false. Components that can be dragged usually should not be picked up/released
   * from a single click event that may have even come from event bubbling. But it can be optionally allowed for some
   * components that have drag functionality but can still be activated with a single click event.
   * (scenery-internal) (part of the scenery listener API)
   */
  public override click( event: SceneryEvent<MouseEvent>, callback?: () => void ): boolean {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener click' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    const success = super.click( event, () => {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener successful press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // notify that we have started a change
      this._start && this._start( event, this as PressedDragListener );

      callback && callback();

      // notify that we have finished a 'drag' activation through click
      this._end && this._end( event, this as PressedDragListener );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    } );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

    return success;
  }

  /**
   * Called when move events are fired on the attached pointer listener during a drag.
   */
  public override drag( event: PressListenerEvent ): void {
    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;

    const point = pressedListener.pointer.point;

    // Ignore global moves that have zero length (Chrome might autofire, see
    // https://code.google.com/p/chromium/issues/detail?id=327114)
    if ( !point || this._globalPoint.equals( point ) ) {
      return;
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener drag' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._dragAction.execute( event );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Attempts to start a touch snag, given a SceneryEvent.
   *
   * Should be safe to be called externally with an event.
   */
  public tryTouchSnag( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener tryTouchSnag' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    if ( this._allowTouchSnag && ( !this.attach || !event.pointer.isAttached() ) ) {
      this.press( event );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Returns a defensive copy of the local-coordinate-frame point of the drag.
   */
  public getGlobalPoint(): Vector2 {
    return this._globalPoint.copy();
  }

  public get globalPoint(): Vector2 { return this.getGlobalPoint(); }

  /**
   * Returns a defensive copy of the local-coordinate-frame point of the drag.
   */
  public getLocalPoint(): Vector2 {
    return this._localPoint.copy();
  }

  public get localPoint(): Vector2 { return this.getLocalPoint(); }

  /**
   * Returns a defensive copy of the parent-coordinate-frame point of the drag.
   */
  public getParentPoint(): Vector2 {
    return this._parentPoint.copy();
  }

  public get parentPoint(): Vector2 { return this.getParentPoint(); }

  /**
   * Returns a defensive copy of the model-coordinate-frame point of the drag.
   */
  public getModelPoint(): Vector2 {
    return this._modelPoint.copy();
  }

  public get modelPoint(): Vector2 { return this.getModelPoint(); }

  /**
   * Returns a defensive copy of the model-coordinate-frame delta.
   */
  public getModelDelta(): Vector2 {
    return this._modelDelta.copy();
  }

  public get modelDelta(): Vector2 { return this.getModelDelta(); }

  /**
   * Maps a point from the global coordinate frame to our drag target's parent coordinate frame.
   *
   * NOTE: This mutates the input vector (for performance)
   *
   * Should be overridden if a custom transformation is needed.
   */
  protected globalToParentPoint( globalPoint: Vector2 ): Vector2 {
    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;

    let referenceResult: Vector2 | undefined;
    if ( assert ) {
      referenceResult = pressedListener.pressedTrail.globalToParentPoint( globalPoint );
    }
    pressedListener.pressedTrail.getParentTransform().getInverse().multiplyVector2( globalPoint );
    assert && assert( globalPoint.equals( referenceResult! ) );
    return globalPoint;
  }

  /**
   * Maps a point from the drag target's parent coordinate frame to its local coordinate frame.
   *
   * NOTE: This mutates the input vector (for performance)
   *
   * Should be overridden if a custom transformation is needed.
   */
  protected parentToLocalPoint( parentPoint: Vector2 ): Vector2 {
    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;

    let referenceResult: Vector2;
    if ( assert ) {
      referenceResult = pressedListener.pressedTrail.lastNode().parentToLocalPoint( parentPoint );
    }
    pressedListener.pressedTrail.lastNode().getTransform().getInverse().multiplyVector2( parentPoint );
    assert && assert( parentPoint.equals( referenceResult! ) );
    return parentPoint;
  }

  /**
   * Maps a point from the drag target's local coordinate frame to its parent coordinate frame.
   *
   * NOTE: This mutates the input vector (for performance)
   *
   * Should be overridden if a custom transformation is needed.
   */
  protected localToParentPoint( localPoint: Vector2 ): Vector2 {
    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;

    let referenceResult: Vector2;
    if ( assert ) {
      referenceResult = pressedListener.pressedTrail.lastNode().localToParentPoint( localPoint );
    }
    pressedListener.pressedTrail.lastNode().getMatrix().multiplyVector2( localPoint );
    assert && assert( localPoint.equals( referenceResult! ) );
    return localPoint;
  }

  /**
   * Maps a point from the drag target's parent coordinate frame to the model coordinate frame.
   *
   * NOTE: This mutates the input vector (for performance)
   *
   * Should be overridden if a custom transformation is needed. Note that by default, unless a transform is provided,
   * the parent coordinate frame will be the same as the model coordinate frame.
   */
  protected parentToModelPoint( parentPoint: Vector2 ): Vector2 {
    if ( this._transform ) {
      const transform = this._transform instanceof Transform3 ? this._transform : this._transform.value;

      transform.getInverse().multiplyVector2( parentPoint );
    }
    return parentPoint;
  }

  /**
   * Maps a point from the model coordinate frame to the drag target's parent coordinate frame.
   *
   * NOTE: This mutates the input vector (for performance)
   *
   * Should be overridden if a custom transformation is needed. Note that by default, unless a transform is provided,
   * the parent coordinate frame will be the same as the model coordinate frame.
   */
  protected modelToParentPoint( modelPoint: Vector2 ): Vector2 {
    if ( this._transform ) {
      const transform = this._transform instanceof Transform3 ? this._transform : this._transform.value;

      transform.getMatrix().multiplyVector2( modelPoint );
    }
    return modelPoint;
  }

  /**
   * Apply a mapping from the drag target's model position to an allowed model position.
   *
   * A common example is using dragBounds, where the position of the drag target is constrained to within a bounding
   * box. This is done by mapping points outside the bounding box to the closest position inside the box. More
   * general mappings can be used.
   *
   * Should be overridden (or use mapPosition) if a custom transformation is needed.
   *
   * @returns - A point in the model coordinate frame
   */
  protected mapModelPoint( modelPoint: Vector2 ): Vector2 {
    if ( this._mapPosition ) {
      return this._mapPosition( modelPoint );
    }
    else if ( this._dragBoundsProperty.value ) {
      return this._dragBoundsProperty.value.closestPointTo( modelPoint );
    }
    else {
      return modelPoint;
    }
  }

  /**
   * Mutates the parentPoint given to account for the initial pointer's offset from the drag target's origin.
   */
  protected applyParentOffset( parentPoint: Vector2 ): void {
    if ( this._offsetPosition ) {
      parentPoint.add( this._offsetPosition( parentPoint, this as PressedDragListener ) );
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
  }

  /**
   * Triggers an update of the drag position, potentially changing position properties.
   *
   * Should be called when something that changes the output positions of the drag occurs (most often, a drag event
   * itself).
   */
  public reposition( globalPoint: Vector2 ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener reposition' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;

    this._globalPoint.set( globalPoint );

    // Update parentPoint mutably.
    this.applyParentOffset( this.globalToParentPoint( this._parentPoint.set( globalPoint ) ) );

    // To compute the delta (new - old), we first mutate it to (-old)
    this._modelDelta.set( this._modelPoint ).negate();

    // Compute the modelPoint from the parentPoint
    this._modelPoint.set( this.mapModelPoint( this.parentToModelPoint( scratchVector2A.set( this._parentPoint ) ) ) );

    // Complete the delta computation
    this._modelDelta.add( this._modelPoint );

    // Apply any mapping changes back to the parent point
    this.modelToParentPoint( this._parentPoint.set( this._modelPoint ) );

    if ( this._translateNode ) {
      pressedListener.pressedTrail.lastNode().translation = this._parentPoint;
    }

    if ( this._positionProperty ) {
      this._positionProperty.value = this._modelPoint.copy(); // Include an extra reference so that it will change.
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'touchenter' events (part of the listener API). (scenery-internal)
   *
   * NOTE: Do not call directly. See the press method instead.
   */
  public touchenter( event: PressListenerEvent ): void {
    if ( event.pointer.isDownProperty.value ) {
      this.tryTouchSnag( event );
    }
  }

  /**
   * Called with 'touchmove' events (part of the listener API). (scenery-internal)
   *
   * NOTE: Do not call directly. See the press method instead.
   */
  public touchmove( event: PressListenerEvent ): void {
    this.tryTouchSnag( event );
  }

  /**
   * Called when an ancestor's transform has changed (when trackAncestors is true).
   */
  private ancestorTransformed(): void {
    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;
    const point = pressedListener.pointer.point;

    if ( point ) {
      // Reposition based on the current point.
      this.reposition( point );
    }
  }

  /**
   * Attaches our transform tracker (begins listening to the ancestor transforms)
   */
  private attachTransformTracker(): void {
    assert && assert( isPressedListener( this ) );
    const pressedListener = this as PressedDragListener;

    if ( this._trackAncestors ) {
      this._transformTracker = new TransformTracker( pressedListener.pressedTrail.copy().removeDescendant() );
      this._transformTracker.addListener( this._transformTrackerListener );
    }
  }

  /**
   * Detaches our transform tracker (stops listening to the ancestor transforms)
   */
  private detachTransformTracker(): void {
    if ( this._transformTracker ) {
      this._transformTracker.removeListener( this._transformTrackerListener );
      this._transformTracker.dispose();
      this._transformTracker = null;
    }
  }

  /**
   * Returns the drag bounds of the listener.
   */
  public getDragBounds(): Bounds2 | null {
    return this._dragBoundsProperty.value;
  }

  public get dragBounds(): Bounds2 | null { return this.getDragBounds(); }

  /**
   * Sets the drag transform of the listener.
   */
  public setTransform( transform: Transform3 | TReadOnlyProperty<Transform3> | null ): void {
    this._transform = transform;
  }

  public set transform( transform: Transform3 | TReadOnlyProperty<Transform3> | null ) { this.setTransform( transform ); }

  public get transform(): Transform3 | TReadOnlyProperty<Transform3> | null { return this.getTransform(); }

  /**
   * Returns the transform of the listener.
   */
  public getTransform(): Transform3 | TReadOnlyProperty<Transform3> | null {
    return this._transform;
  }

  /**
   * Interrupts the listener, releasing it (canceling behavior).
   *
   * This effectively releases/ends the press, and sets the `interrupted` flag to true while firing these events
   * so that code can determine whether a release/end happened naturally, or was canceled in some way.
   *
   * This can be called manually, but can also be called through node.interruptSubtreeInput().
   */
  public override interrupt(): void {
    if ( this.pointer && this.pointer.isTouchLike() ) {
      this._lastInterruptedTouchLikePointer = this.pointer;
    }

    super.interrupt();
  }

  /**
   * Returns whether a press can be started with a particular event.
   */
  public override canPress( event: PressListenerEvent ): boolean {
    if ( event.pointer === this._lastInterruptedTouchLikePointer ) {
      return false;
    }

    return super.canPress( event );
  }

  /**
   * Disposes the listener, releasing references. It should not be used after this.
   */
  public override dispose(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener dispose' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._dragAction.dispose();

    this.detachTransformTracker();

    super.dispose();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Creates an input listener that forwards events to the specified input listener. The target listener should
   * probably be using PressListener.options.targetNode so that the forwarded drag has the correct Trail
   *
   * See https://github.com/phetsims/scenery/issues/639
   */
  public static createForwardingListener( down: ( event: PressListenerEvent ) => void,
                                          providedOptions?: CreateForwardingListenerOptions ): TInputListener {

    const options = optionize<CreateForwardingListenerOptions, CreateForwardingListenerOptions>()( {
      allowTouchSnag: true // see https://github.com/phetsims/scenery/issues/999
    }, providedOptions );

    return {
      down( event ) {
        if ( event.canStartPress() ) {
          down( event );
        }
      },
      touchenter( event ) {
        options.allowTouchSnag && this.down!( event );
      },
      touchmove( event ) {
        options.allowTouchSnag && this.down!( event );
      }
    };
  }
}

scenery.register( 'DragListener', DragListener );

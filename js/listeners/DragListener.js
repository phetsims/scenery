// Copyright 2013-2017, University of Colorado Boulder

/**
 * PressListener subtype customized for handling most drag-related listener needs.
 *
 * TODO: doc the coordinate frames (global, parent, local, model)
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

    assert && assert( !options.mapLocation || !options.dragBounds,
      'mapLocation and dragBounds cannot both be provided, as they both handle mapping of the drag point' );

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
    // TODO: Handle specifying the pressed trail, so this is done accurately?
    // Most general: pass in a predicate to identify the last node in the trail (if we have a trail)
    globalToParentPoint: function( globalPoint ) {
      return this.pressedTrail.globalToParentPoint( globalPoint );
    },

    parentToLocalPoint: function( parentPoint ) {
      // TODO: test scale/rotation on our dragged thing
      return this.pressedTrail.lastNode().parentToLocalPoint( parentPoint );
    },

    localToParentPoint: function( localPoint ) {
      // TODO: test scale/rotation on our dragged thing
      return this.pressedTrail.lastNode().localToParentPoint( localPoint );
    },

    parentToModelPoint: function( parentPoint ) {
      // TODO: override
      if ( this._transform ) {
        return this._transform.inversePosition2( parentPoint );
      }
      else {
        return parentPoint;
      }
    },

    modelToParentPoint: function( modelPoint ) {
      // TODO: override
      if ( this._transform ) {
        return this._transform.transformPosition2( modelPoint );
      }
      else {
        return modelPoint;
      }
    },

    // mark as overrideable
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

    press: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      var success = PressListener.prototype.press.call( this, event ); // TODO: do we need to delay notification with options release?

      if ( success ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener successful press' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.attachTransformTracker();

        // TODO: scratch vectors
        this.initialLocalPoint = this.parentToLocalPoint( this.globalToParentPoint( this.pointer.point ) );

        this.reposition( this.pointer.point );

        this._start && this._start( event );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

      return success;
    },

    release: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener release' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      PressListener.prototype.release.call( this );

      this.detachTransformTracker();

      this._end && this._end();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    drag: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener drag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // NOTE: This is done first, before the drag listener is called
      this.reposition( this.pointer.point );

      //TODO ignore global moves that have zero length (Chrome might autofire, see https://code.google.com/p/chromium/issues/detail?id=327114)
      //TODO: should this apply in PressListener's drag?
      PressListener.prototype.drag.call( this, event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    // TODO: hardcode pointer.point?
    reposition: function( globalPoint ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener reposition' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // TODO: scratch vectors, better codepath for minimizing computation
      var parentPointerPoint = this.globalToParentPoint( globalPoint );
      var parentLocalPoint = this.localToParentPoint( this.initialLocalPoint );
      var parentOriginPoint = this.localToParentPoint( new Vector2() ); // usually node.translation
      var parentOffset = parentOriginPoint.minus( parentLocalPoint );
      this.parentPoint = this._applyOffset ? parentPointerPoint.plus( parentOffset ) : parentPointerPoint;
      this.modelPoint = this.mapModelPoint( this.parentToModelPoint( this.parentPoint ) );
      this.parentPoint = this.modelToParentPoint( this.modelPoint ); // apply any mapping changes

      if ( this._translateNode ) {
        this.pressedTrail.lastNode().translation = this.parentPoint;
      }

      if ( this._locationProperty ) {
        this._locationProperty.value = this.modelPoint;
      }

      // TODO: consider other options to handle here

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    ancestorTransformed: function() {
      this.reposition( this.pointer.point );
    },

    touchenter: function( event ) {
      this.tryTouchSnag( event );
    },

    touchmove: function( event ) {
      this.tryTouchSnag( event );
    },

    tryTouchSnag: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DragListener tryTouchSnag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      if ( this._allowTouchSnag && !event.pointer.isAttached() ) {
        this.press( event );
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
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

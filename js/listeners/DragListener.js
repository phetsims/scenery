// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO: doc
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );
  var PressListener = require( 'SCENERY/listeners/PressListener' );

  /**
   * TODO: doc
   */
  function DragListener( options ) {
    options = _.extend( {
      allowTouchSnag: false, // TODO: decide on appropriate default
      translateNode: false,
      transform: null,
      locationProperty: null, // TODO doc
      mapLocation: null, // TODO: doc
      dragBounds: null
    }, options );

    assert && assert( !options.mapLocation || !options.dragBounds,
      'mapLocation and dragBounds cannot both be provided, as they both handle mapping of the drag point' );

    PressListener.call( this, options );

    this._allowTouchSnag = options.allowTouchSnag;
    this._translateNode = options.translateNode;
    this._transform = options.transform;
    this._locationProperty = options.locationProperty;
    this._mapLocation = options.mapLocation;
    this._dragBounds = options.dragBounds;

    this._initialLocalPoint = null;
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
      PressListener.prototype.press.call( this, event ); // TODO: do we need to delay notification with options release?

      // TODO: scratch vectors
      this._initialLocalPoint = this.parentToLocalPoint( this.globalToParentPoint( this.pointer.point ) );

      this.reposition( this.pointer.point );
    },

    drag: function( event ) {
      //TODO ignore global moves that have zero length (Chrome might autofire, see https://code.google.com/p/chromium/issues/detail?id=327114)
      //TODO: should this apply in PressListener's drag?
      PressListener.prototype.drag.call( this, event );

      this.reposition( this.pointer.point );
    },

    // TODO: hardcode pointer.point?
    reposition: function( globalPoint ) {
      // TODO: scratch vectors, better codepath for minimizing computation
      var parentPointerPoint = this.globalToParentPoint( globalPoint );
      var parentLocalPoint = this.localToParentPoint( this._initialLocalPoint );
      var parentOriginPoint = this.localToParentPoint( new Vector2() ); // usually node.translation
      var parentOffset = parentOriginPoint.minus( parentLocalPoint );
      var parentPoint = parentPointerPoint.plus( parentOffset );
      var modelPoint = this.mapModelPoint( this.parentToModelPoint( parentPoint ) );
      parentPoint = this.modelToParentPoint( modelPoint ); // apply any mapping changes

      if ( this._translateNode ) {
        this.pressedTrail.lastNode().translation = parentPoint;
      }

      if ( this._locationProperty ) {
        this._locationProperty.value = modelPoint;
      }
    },

    touchenter: function( event ) {
      this.tryTouchSnag( event );
    },

    touchmove: function( event ) {
      this.tryTouchSnag( event );
    },

    tryTouchSnag: function( event ) {
      if ( this._allowTouchSnag ) {
        this.tryPress( event );
      }
    }

  } );

  return DragListener;
} );

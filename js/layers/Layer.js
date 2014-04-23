// Copyright 2002-2014, University of Colorado

/**
 * Base code for layers that helps with shared layer functions
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  var globalIdCounter = 1;
  
  /*
   * Required arguments:
   * $main     - the jQuery-wrapped container for the scene
   * scene     - the scene itself
   * baseNode  - the base node for this layer
   */
  scenery.Layer = function Layer( args ) {
    
    // assign a unique ID to this layer
    this._id = globalIdCounter++;
    
    this.$main = args.$main;
    this.scene = args.scene;
    this.baseNode = args.baseNode;
    
    // TODO: cleanup of flags!
    this.usesPartialCSSTransforms = args.cssTranslation || args.cssRotation || args.cssScale;
    this.cssTranslation = args.cssTranslation; // CSS for the translation
    this.cssRotation = args.cssRotation;       // CSS for the rotation
    this.cssScale = args.cssScale;             // CSS for the scaling
    this.cssTransform = args.cssTransform;     // CSS for the entire base node (will ignore other partial transforms)
    assert && assert( !( this.usesPartialCSSTransforms && this.cssTransform ), 'Do not specify both partial and complete CSS transform arguments.' );
    
    // initialize to fully dirty so we draw everything the first time
    // bounds in global coordinate frame
    this.dirtyBounds = Bounds2.EVERYTHING;
    
    this.setStartBoundary( args.startBoundary );
    this.setEndBoundary( args.endBoundary );
    
    // set baseTrail from the scene to our baseNode
    if ( this.baseNode === this.scene ) {
      this.baseTrail = new scenery.Trail( this.scene );
    } else {
      this.baseTrail = this.startPaintedTrail.upToNode( this.baseNode );
      assert && assert( this.baseTrail.lastNode() === this.baseNode );
    }
    
    // we reference all painted trails in an unordered way
    this._layerTrails = []; // TODO: performance: remove layerTrails if possible!
    this._instanceCount = 0; // track how many instances we are tracking (updated in stitching by instances)
    
    var layer = this;
    
    // whenever the base node's children or self change bounds, signal this. we want to explicitly ignore the base node's main bounds for
    // CSS transforms, since the self / children bounds may not have changed
    this.baseNodeBoundsListener = function() {
      layer.baseNodeInternalBoundsChange(); // TODO: verify that this is working as expected
    };
    this.baseNode.addEventListener( 'localBounds', this.baseNodeBoundsListener );
    
    this.fitToBounds = this.usesPartialCSSTransforms || this.cssTransform;
    assert && assert( this.fitToBounds || this.baseNode === this.scene, 'If the baseNode is not the scene, we need to fit the bounds' );
    
    // used for CSS transforms where we need to transform our base node's bounds into the (0,0,w,h) bounds range
    this.baseNodeTransform = new Transform3();
    //this.baseNodeInteralBounds = Bounds2.NOTHING; // stores the bounds transformed into (0,0,w,h)
    
    this.disposed = false; // track whether we have been disposed or not
  };
  var Layer = scenery.Layer;
  
  Layer.prototype = {
    constructor: Layer,
    
    setStartBoundary: function( boundary ) {
      // console.log( 'setting start boundary on layer ' + this.getId() + ': ' + boundary.toString() );
      this.startBoundary = boundary;
      
      // TODO: deprecate these, use boundary references instead? or boundary convenience functions
      this.startPaintedTrail = this.startBoundary.nextPaintedTrail;
      
      // set immutability guarantees
      this.startPaintedTrail.setImmutable();
    },
    
    setEndBoundary: function( boundary ) {
      // console.log( 'setting end boundary on layer ' + this.getId() + ': ' + boundary.toString() );
      this.endBoundary = boundary;
      
      // TODO: deprecate these, use boundary references instead? or boundary convenience functions
      this.endPaintedTrail = this.endBoundary.previousPaintedTrail;
      
      // set immutability guarantees
      this.endPaintedTrail.setImmutable();
    },
    
    toString: function() {
      return this.getName() + ' ' + ( this.startPaintedTrail ? this.startPaintedTrail.toString() : '!' ) + ' => ' + ( this.endPaintedTrail ? this.endPaintedTrail.toString() : '!' );
    },
    
    getId: function() {
      return this._id;
    },
    get id() { return this._id; }, // ES5 version
    
    // painted trails associated with the layer, NOT necessarily in order
    getLayerTrails: function() {
      return this._layerTrails.slice( 0 );
    },
    
    getPaintedTrailCount: function() {
      return this._layerTrails.length;
    },
    
    /*---------------------------------------------------------------------------*
    * Abstract
    *----------------------------------------------------------------------------*/
    
    render: function( state ) {
      throw new Error( 'Layer.render unimplemented' );
    },
    
    // TODO: consider a stack-based model for transforms?
    // TODO: is this necessary? verify with the render state
    applyTransformationMatrix: function( matrix ) {
      throw new Error( 'Layer.applyTransformationMatrix unimplemented' );
    },
    
    // adds a trail (with the last node) to the layer
    addInstance: function( instance ) {
      var trail = instance.trail;
      
      if ( assert ) {
        _.each( this._layerTrails, function( otherTrail ) {
          assert( !trail.equals( otherTrail ), 'trail in addInstance should not already exist in a layer' );
        } );
      }
      
      // TODO: sync this with DOMLayer's implementation
      this._layerTrails.push( trail );
      trail.setImmutable(); // don't allow this Trail to be changed
    },
    
    // removes a trail (with the last node) to the layer
    removeInstance: function( instance ) {
      // TODO: sync this with DOMLayer's implementation
      var i;
      for ( i = 0; i < this._layerTrails.length; i++ ) {
        this._layerTrails[i].reindex();
        if ( this._layerTrails[i].compare( instance.trail ) === 0 ) {
          break;
        }
      }
      assert && assert( i < this._layerTrails.length );
      
      this._layerTrails.splice( i, 1 );
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      this.startBoundary.reindex();
      this.endBoundary.reindex();
    },
    
    pushClipShape: function( shape ) {
      throw new Error( 'Layer.pushClipShape unimplemented' );
    },
    
    popClipShape: function() {
      throw new Error( 'Layer.popClipShape unimplemented' );
    },
    
    renderToCanvas: function( canvas, context, delayCounts ) {
      throw new Error( 'Layer.renderToCanvas unimplemented' );
    },
    
    dispose: function() {
      assert && assert( !this.disposed, 'Layer has already been disposed!' );
      
      this.disposed = true;
      
      // clean up listeners
      this.baseNode.removeEventListener( 'localBounds', this.baseNodeBoundsListener );
    },
    
    getName: function() {
      throw new Error( 'Layer.getName unimplemented' );
    },
    
    // called when the base node's "internal" (self or child) bounds change, but not when it is just from the base node's own transform changing
    baseNodeInternalBoundsChange: function() {
      // no error, many times this doesn't need to be handled
    }
    
  };
  
  Layer.cssTransformPadding = 3;
  
  return Layer;
} );



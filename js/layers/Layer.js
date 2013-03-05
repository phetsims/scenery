// Copyright 2002-2012, University of Colorado

/**
 * Base code for layers that helps with shared layer functions
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', true );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  /*
   * Required arguments:
   * $main     - the jQuery-wrapped container for the scene
   * scene     - the scene itself
   * baseNode  - the base node for this layer
   *
   * Optional arguments:
   * batchDOMChanges: false - Only run DOM manipulation from within requestAnimationFrame calls
   */
  scenery.Layer = function( args ) {
    this.$main = args.$main;
    this.scene = args.scene;
    this.baseNode = args.baseNode;
    
    // DOM batching
    this.batchDOMChanges = args.batchDOMChanges || false;
    this.pendingDOMChanges = [];
    this.applyingDOMChanges = false;
    
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
    
    this.startBoundary = args.startBoundary;
    this.endBoundary = args.endBoundary;
    
    // TODO: deprecate these, use boundary references instead? or boundary convenience functions
    this.startPointer = this.startBoundary.nextStartPointer;
    this.endPointer = this.endBoundary.previousEndPointer;
    this.startSelfTrail = this.startBoundary.nextSelfTrail;
    this.endSelfTrail = this.endBoundary.previousSelfTrail;
    
    // set baseTrail from the scene to our baseNode
    if ( this.baseNode === this.scene ) {
      this.baseTrail = new scenery.Trail( this.scene );
    } else {
      this.baseTrail = this.startPointer.trail.copy();
      assert && assert( this.baseTrail.lastNode() === this.baseNode );
    }
    
    var layer = this;
    
    // whenever the base node's children or self change bounds, signal this. we want to explicitly ignore the base node's main bounds for
    // CSS transforms, since the self / children bounds may not have changed
    this.baseNodeListener = {
      selfBounds: function( bounds ) {
        layer.baseNodeInternalBoundsChange();
      },
      
      childBounds: function( bounds ) {
        layer.baseNodeInternalBoundsChange();
      }
    };
    this.baseNode.addEventListener( this.baseNodeListener );
    
    this.fitToBounds = this.usesPartialCSSTransforms || this.cssTransform;
    assert && assert( this.fitToBounds || this.baseNode === this.scene, 'If the baseNode is not the scene, we need to fit the bounds' );
    
    // used for CSS transforms where we need to transform our base node's bounds into the (0,0,w,h) bounds range
    this.baseNodeTransform = new Transform3();
    //this.baseNodeInteralBounds = Bounds2.NOTHING; // stores the bounds transformed into (0,0,w,h)
  };
  var Layer = scenery.Layer;
  
  Layer.prototype = {
    constructor: Layer,
    
    getStartPointer: function() {
      return this.startPointer;
    },
    
    getEndPointer: function() {
      return this.endPointer;
    },
    
    flushDOMChanges: function() {
      // signal that we are now applying the changes, so calling domChange will trigger instant evaluation
      this.applyingDOMChanges = true;
      
      // TODO: consider a 'try' block, as things may now not exist? ideally we should only batch things that will always work
      _.each( this.pendingDOMChanges, function( change ) {
        change();
      } );
      
      // removes all entries
      this.pendingDOMChanges.splice( 0, this.pendingDOMChanges.length );
      
      // start batching again
      this.applyingDOMChanges = false;
    },
    
    domChange: function( callback ) {
      if ( this.batchDOMChanges && !this.applyingDOMChanges ) {
        this.pendingDOMChanges.push( callback );
      } else {
        callback();
      }
    },
    
    toString: function() {
      return this.getName() + ' ' + ( this.startPointer ? this.startPointer.toString() : '!' ) + ' (' + ( this.startSelfTrail ? this.startSelfTrail.toString() : '!' ) + ') => ' + ( this.endPointer ? this.endPointer.toString() : '!' ) + ' (' + ( this.endSelfTrail ? this.endSelfTrail.toString() : '!' ) + ')';
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
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      throw new Error( 'unimplemented layer reindex' );
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
      this.baseNode.removeEventListener( this.baseNodeListener );
    },
    
    // args should contain node, bounds (local bounds), transform, trail
    markDirtyRegion: function( args ) {
      throw new Error( 'Layer.markDirtyRegion unimplemented' );
    },
    
    // args should contain node, type (append, prepend, set), matrix, transform, trail
    transformChange: function( args ) {
      throw new Error( 'Layer.transformChange unimplemented' );
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



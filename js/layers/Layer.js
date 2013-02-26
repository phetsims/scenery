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
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/Trail' );
  
  /*
   * Typical arguments:
   * $main     - the jQuery-wrapped container for the scene
   * scene     - the scene itself
   * baseNode  - the base node for this layer
   */
  scenery.Layer = function( args, entry ) {
    this.$main = args.$main;
    this.scene = args.scene;
    this.baseNode = args.baseNode;
    
    // initialize to fully dirty so we draw everything the first time
    // bounds in global coordinate frame
    this.dirtyBounds = Bounds2.EVERYTHING;
    
    this.startPointer = entry.startPointer;
    this.endPointer = entry.endPointer;
    this.startSelfTrail = entry.startSelfTrail;
    this.endSelfTrail = entry.endSelfTrail;
    
    // set baseTrail from the scene to our baseNode
    if ( this.baseNode === this.scene ) {
      this.baseTrail = new scenery.Trail( this.scene );
    } else {
      assertExtra && assertExtra( _.contains( this.startPointer.trail.nodes, this.baseNode ) );
      assertExtra && assertExtra( _.contains( this.endPointer.trail.nodes, this.baseNode ) );
      this.baseTrail = this.startPointer.trail.subtrailTo( this.baseNode );
    }
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
      throw new Error( 'Layer.dispose unimplemented' );
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
    }
  };
  
  return Layer;
} );



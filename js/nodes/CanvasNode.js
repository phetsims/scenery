// Copyright 2002-2014, University of Colorado

/**
 * A node that can be custom-drawn with Canvas calls. Manual handling of dirty region repainting.
 *
 * setCanvasBounds (or the mutator canvasBounds) should be used to set the area that is drawn to (otherwise nothing
 * will show up)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' );
  require( 'SCENERY/display/Renderer' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  
  // pass a canvasBounds option if you want to specify the self bounds
  scenery.CanvasNode = function CanvasNode( options ) {
    Node.call( this, options );
    this.setRendererBitmask( scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas );
    
    if ( options && options.canvasBounds ) {
      this.setCanvasBounds( options.canvasBounds );
    }
  };
  var CanvasNode = scenery.CanvasNode;
  
  inherit( Node, CanvasNode, {
    
    // how to set the bounds of the CanvasNode
    setCanvasBounds: function( selfBounds ) {
      this.invalidateSelf( selfBounds );
    },
    set canvasBounds( value ) { this.setCanvasBounds( value ); },
    get canvasBounds() { return this.getSelfBounds(); },
    
    isPainted: function() {
      return true;
    },
    
    // override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path
    paintCanvas: function( wrapper ) {
      throw new Error( 'CanvasNode needs paintCanvas implemented' );
    },
    
    invalidatePaint: function() {
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[i].markDirty();
      }
    },
    
    canvasPaintSelf: function( wrapper ) {
      this.paintCanvas( wrapper );
    },
    
    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      return false;
      // throw new Error( 'CanvasNode needs containsPointSelf implemented' );
    },
    
    createCanvasDrawable: function( renderer, instance ) {
      return CanvasNode.CanvasNodeDrawable.createFromPool( renderer, instance );
    },
    
    // whether this node's self intersects the specified bounds, in the local coordinate frame
    // intersectsBoundsSelf: function( bounds ) {
    //   // TODO: implement?
    // },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.CanvasNode( {' + propLines + '} )'; // TODO: no real way to do this nicely?
    }
    
  } );
  
  CanvasNode.prototype._mutatorKeys = [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys );
  
  /*---------------------------------------------------------------------------*
  * Canvas rendering
  *----------------------------------------------------------------------------*/
  
  CanvasNode.CanvasNodeDrawable = CanvasSelfDrawable.createDrawable( {
    type: function CanvasNodeDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasNode( wrapper, node ) {
      if ( !node._selfBounds.isEmpty() ) {
        node.paintCanvas( wrapper );
      }
    },
    usesFill: false,
    usesStroke: false
  } );
  
  return CanvasNode;
} );



// Copyright 2002-2014, University of Colorado Boulder

/**
 * A node that can be custom-drawn with WebGL calls. Manual handling of dirty region repainting.  Analogous to WebGLNode
 *
 * setCanvasBounds (or the mutator canvasBounds) should be used to set the area that is drawn to (otherwise nothing
 * will show up)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' );
  require( 'SCENERY/display/Renderer' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  // pass a canvasBounds option if you want to specify the self bounds
  scenery.WebGLNode = function WebGLNode( options ) {
    Node.call( this, options );
    this.setRendererBitmask( scenery.bitmaskBoundsValid | scenery.bitmaskSupportsWebGL );
  };
  var WebGLNode = scenery.WebGLNode;

  inherit( Node, WebGLNode, {

    // how to set the bounds of the WebGLNode
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
      throw new Error( 'WebGLNode needs paintCanvas implemented' );
    },

    invalidatePaint: function() {
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirty();
      }
    },

    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      return false;
      // throw new Error( 'WebGLNode needs containsPointSelf implemented' );
    },

    canvasPaintSelf: function( wrapper ) {
      assert && assert( 'unimplemented: canvasPaintSelf in WebGLNode' );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return WebGLNode.WebGLNodeDrawable.createFromPool( renderer, instance );
    },

    // whether this node's self intersects the specified bounds, in the local coordinate frame
    // intersectsBoundsSelf: function( bounds ) {
    //   // TODO: implement?
    // },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.WebGLNode( {' + propLines + '} )'; // TODO: no real way to do this nicely?
    }

  } );

  WebGLNode.prototype._mutatorKeys = [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys );

  WebGLNode.WebGLNodeDrawable = inherit( WebGLSelfDrawable, function WebGLNodeDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called either from the constructor, or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );
    },

    initializeContext: function( gl ) {
      this.gl = gl;

      //this.node.initializeContext( gl );//TODO: Rename call to initializeContext?  Breaks with 0.1 but should be done for consistency.
    },

    render: function( shaderProgram ) {
      this.node.render( this.gl, shaderProgram );
    },

    dispose: function() {
      if ( this.gl ) {
        this.node.dispose();
        this.gl = null;
      }

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    onAttach: function( node ) {

    },

    // release the drawable
    onDetach: function( node ) {
      //OHTWO TODO: are we missing the disposal?
    },

    // forward call to the WebGLNode
    get shaderAttributes() {
      return this.node.shaderAttributes;
    },

    update: function() {
      this.dirty = false;

      if ( this.paintDirty ) {
        this.updateRectangle();

        this.setToCleanState();
      }
    }
  } );

  // set up pooling
  SelfDrawable.Poolable.mixin( WebGLNode.WebGLNodeDrawable );

  return WebGLNode;
} );

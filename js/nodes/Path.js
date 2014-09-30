// Copyright 2002-2014, University of Colorado Boulder

/**
 * A Path draws a Shape with a specific type of fill and stroke.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Shape = require( 'KITE/Shape' );

  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  require( 'SCENERY/layers/Renderer' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );

  scenery.Path = function Path( shape, options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    // NOTE: _shape can be lazily constructed, in the case of types like Rectangle where they have their own drawing code
    this._shape = null;
    this._strokedShape = null; // a stroked copy of the shape, lazily computed

    // ensure we have a parameter object
    options = options || {};

    this.initializeFillable();
    this.initializeStrokable();

    Node.call( this );
    this.invalidateSupportedRenderers();
    this.setShape( shape );
    this.mutate( options );
  };
  var Path = scenery.Path;

  inherit( Node, Path, {
    // allow more specific path types (Rectangle, Line) to override what restrictions we have
    getPathRendererBitmask: function() {
      return scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG;
    },

    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getPathRendererBitmask() );
    },

    // sets the shape drawn, or null to remove the shape
    setShape: function( shape ) {
      if ( this._shape !== shape ) {
        if ( typeof shape === 'string' ) {
          // be content with setShape always invalidating the shape?
          shape = new Shape( shape );
        }
        this._shape = shape;
        this.invalidateShape();
      }
      return this;
    },

    getShape: function() {
      return this._shape;
    },

    getStrokedShape: function() {
      if ( !this._strokedShape ) {
        this._strokedShape = this.getShape().getStrokedShape( this._lineDrawingStyles );
      }
      return this._strokedShape;
    },

    invalidateShape: function() {
      this.markOldSelfPaint();

      this._strokedShape = null;

      if ( this.hasShape() ) {
        this.invalidateSelf( this.computeShapeBounds() );
        this.invalidatePaint();
      }
    },

    // separated out, so that we can override this with a faster version in subtypes. includes the Stroke, if any
    computeShapeBounds: function() {
      return this._stroke ? this.getStrokedShape().bounds : this.getShape().bounds;
    },

    // @override
    getTransformedSelfBounds: function( matrix ) {
      return ( this._stroke ? this.getStrokedShape() : this.getShape() ).getBoundsWithTransform( matrix );
    },

    // hook stroke mixin changes to invalidation
    invalidateStroke: function() {
      this.invalidateShape();
    },

    hasShape: function() {
      return this._shape;
    },

    paintCanvas: function( wrapper ) {
      var context = wrapper.context;

      if ( this.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        this._shape.writeToContext( context );

        if ( this._fill ) {
          this.beforeCanvasFill( wrapper ); // defined in Fillable
          context.fill();
          this.afterCanvasFill( wrapper ); // defined in Fillable
        }
        if ( this._stroke ) {
          this.beforeCanvasStroke( wrapper ); // defined in Strokable
          context.stroke();
          this.afterCanvasStroke( wrapper ); // defined in Strokable
        }
      }
    },

    paintWebGL: function( state ) {
      throw new Error( 'Path.prototype.paintWebGL unimplemented' );
    },

    // svg element, the <defs> block, and the associated group for this node's transform
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( scenery.svgns, 'path' );
    },

    updateSVGFragment: function( path ) {
      var svgPath = this.hasShape() ? this._shape.getSVGPath() : "";

      // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
      // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
      // the attribute can cause this bug
      if ( !svgPath ) { svgPath = 'M0 0'; }

      if ( svgPath ) {
        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      }
      else if ( path.hasAttribute( 'd' ) ) {
        path.removeAttribute( 'd' );
      }

      path.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },

    // support patterns, gradients, and anything else we need to put in the <defs> block
    updateSVGDefs: function( svg, defs ) {
      // remove old definitions if they exist
      this.removeSVGDefs( svg, defs );

      // add new ones if applicable
      this.addSVGFillDef( svg, defs );
      this.addSVGStrokeDef( svg, defs );

      assert && assert( !this.requiresSVGBoundsWorkaround(), 'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );
    },

    // cleans up references created with udpateSVGDefs()
    removeSVGDefs: function( svg, defs ) {
      this.removeSVGFillDef( svg, defs );
      this.removeSVGStrokeDef( svg, defs );
    },

    isPainted: function() {
      return true;
    },

    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      var result = false;
      if ( !this.hasShape() ) {
        return result;
      }

      // if this node is fillPickable, we will return true if the point is inside our fill area
      if ( this._fillPickable ) {
        result = this.getShape().containsPoint( point );
      }

      // also include the stroked region in the hit area if strokePickable
      if ( !result && this._strokePickable ) {
        result = this.getStrokedShape().containsPoint( point );
      }
      return result;
    },

    // whether this node's self intersects the specified bounds, in the local coordinate frame
    intersectsBoundsSelf: function( bounds ) {
      // TODO: should a shape's stroke be included?
      return this.hasShape() ? this._shape.intersectsBounds( bounds ) : false;
    },

    set shape( value ) { this.setShape( value ); },
    get shape() { return this.getShape(); },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Path( ' + ( this._shape ? this._shape.toString() : this._shape ) + ', {' + propLines + '} )';
    },

    getPropString: function( spaces, includeChildren ) {
      var result = Node.prototype.getPropString.call( this, spaces, includeChildren );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      return result;
    }
  } );

  Path.prototype._mutatorKeys = [ 'shape' ].concat( Node.prototype._mutatorKeys );

  // mix in fill/stroke handling code. for now, this is done after 'shape' is added to the mutatorKeys so that stroke parameters
  // get set first
  /* jshint -W064 */
  Fillable( Path );
  Strokable( Path );

  return Path;
} );



// Copyright 2002-2014, University of Colorado

/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );

  scenery.CanvasSelfDrawable = function CanvasSelfDrawable( renderer, instance ) {
    this.initializeCanvasSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  };
  var CanvasSelfDrawable = scenery.CanvasSelfDrawable;

  inherit( SelfDrawable, CanvasSelfDrawable, {
    initializeCanvasSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      // this is the same across lifecycles
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

      instance.addRelativeTransformListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
      instance.addRelativeTransformPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated

      return this;
    },

    markTransformDirty: function() {
      this.markDirty();
    },

    dispose: function() {
      this.instance.removeRelativeTransformListener( this.transformListener );
      this.instance.removeRelativeTransformPrecompute();

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  // methods for forwarding dirty messages
  function canvasSelfDirty() {
    // we pass this method and it is only called with blah.call( ... ), where the 'this' reference is set. ignore jshint
    /* jshint -W040 */
    this.markDirty();
  }

  // options takes: type, paintCanvas( wrapper ), usesFill, usesStroke, and dirtyMethods (array of string names of methods that make the state dirty)
  CanvasSelfDrawable.createDrawable = function( options ) {
    var type = options.type;
    var paintCanvas = options.paintCanvas;
    var usesFill = options.usesFill;
    var usesStroke = options.usesStroke;

    assert && assert( typeof type === 'function' );
    assert && assert( typeof paintCanvas === 'function' );
    assert && assert( typeof usesFill === 'boolean' );
    assert && assert( typeof usesStroke === 'boolean' );

    inherit( CanvasSelfDrawable, type, {
      initialize: function( renderer, instance ) {
        this.initializeCanvasSelfDrawable( renderer, instance );

        return this; // allow for chaining
      },

      // general flag set on the state, which we forward directly to the drawable's paint flag
      markPaintDirty: function() {
        this.markDirty();
      },

      paintCanvas: paintCanvas,

      onAttach: function( node ) {

      },

      // release the drawable
      onDetach: function( node ) {
        //OHTWO TODO: are we missing the disposal?
      },

      update: function() {
        this.dirty = false;
      }
    } );

    // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for Canvas (for now)
    if ( usesFill ) {
      /* jshint -W064 */
      Fillable.FillableStateless( type );
    }
    if ( usesStroke ) {
      /* jshint -W064 */
      Strokable.StrokableStateless( type );
    }

    // set up pooling
    /* jshint -W064 */
    SelfDrawable.Poolable( type );

    if ( options.dirtyMethods ) {
      for ( var i = 0; i < options.dirtyMethods.length; i++ ) {
        type.prototype[options.dirtyMethods[i]] = canvasSelfDirty;
      }
    }

    return type;
  };

  return CanvasSelfDrawable;
} );

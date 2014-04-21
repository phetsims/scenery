// Copyright 2002-2013, University of Colorado

/**
 * TODO docs
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
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
    }
  } );
  
  // options takes: type, paintCanvas( wrapper ), usesFill, usesStroke
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
      
      onAttach: function( node ) {
        
      },
      
      paintCanvas: paintCanvas,
      
      // release the drawable
      onDetach: function( node ) {
        // put us back in the pool
        this.freeToPool();
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
    
    return type;
  };
  
  return CanvasSelfDrawable;
} );

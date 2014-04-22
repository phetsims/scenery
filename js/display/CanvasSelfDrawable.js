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
      
      onAttach: function( node ) {
        
      },
      
      // general flag set on the state, which we forward directly to the drawable's paint flag
      markPaintDirty: function() {
        this.markDirty();
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
    
    if ( options.dirtyMethods ) {
      for ( var i = 0; i < options.dirtyMethods.length; i++ ) {
        type.prototype[options.dirtyMethods[i]] = canvasSelfDirty;
      }
    }
    
    return type;
  };
  
  return CanvasSelfDrawable;
} );

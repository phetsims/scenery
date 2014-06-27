// Copyright 2002-2014, University of Colorado

/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying any changes at a later time.
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

  scenery.SVGSelfDrawable = function SVGSelfDrawable( renderer, instance ) {
    this.initializeSVGSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  };
  var SVGSelfDrawable = scenery.SVGSelfDrawable;

  inherit( SelfDrawable, SVGSelfDrawable, {
    initializeSVGSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      this.svgElement = null; // should be filled in by subtype
      this.defs = null; // will be updated by updateDefs()

      return this;
    },

    // @public: called when the defs block changes
    // NOTE: should generally be overridden by drawable subtypes, so they can apply their defs changes
    updateDefs: function( defs ) {
      this.defs = defs;
    },

    // @public: called from elsewhere to update the SVG element
    update: function() {
      if ( this.dirty ) {
        this.dirty = false;
        this.updateSVG();
      }
    },

    // @protected: called to update the visual appearance of our svgElement
    updateSVG: function() {
      // should generally be overridden by drawable subtypes to implement the update
    },

    dispose: function() {
      this.defs = null;

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  /*
   * Options contains:
   *   type - the constructor, should be of the form: function SomethingSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); }.
   *          Used for debugging constructor name.
   *   stateType - function to apply to mix-in the state (TODO docs)
   *   initialize( renderer, instance ) - should initialize this.svgElement if it doesn't already exist, and set up any other initial state properties
   *   updateSVG() - updates the svgElement to the latest state recorded
   *   updateDefs( defs ) - called when the SVG <defs> object needs to be switched (or initialized)
   *   usesFill - whether we include fillable state & defs
   *   usesStroke - whether we include strokable state & defs
   *   keepElements - when disposing a drawable (not used anymore), should we keep a reference to the SVG element so we don't have to recreate it when reinitialized?
   */
  SVGSelfDrawable.createDrawable = function( options ) {
    var type = options.type;
    var stateType = options.stateType;
    var initializeSelf = options.initialize;
    var updateSVGSelf = options.updateSVG;
    var updateDefsSelf = options.updateDefs;
    var usesFill = options.usesFill;
    var usesStroke = options.usesStroke;
    var keepElements = options.keepElements;

    assert && assert( typeof type === 'function' );
    assert && assert( typeof stateType === 'function' );
    assert && assert( typeof initializeSelf === 'function' );
    assert && assert( typeof updateSVGSelf === 'function' );
    assert && assert( !updateDefsSelf || ( typeof updateDefsSelf === 'function' ) );
    assert && assert( typeof usesFill === 'boolean' );
    assert && assert( typeof usesStroke === 'boolean' );
    assert && assert( typeof keepElements === 'boolean' );

    inherit( SVGSelfDrawable, type, {
      initialize: function( renderer, instance ) {
        this.initializeSVGSelfDrawable( renderer, instance );
        this.initializeState(); // assumes we have a state mixin

        initializeSelf.call( this, renderer, instance );

        // tracks our current defs object, so we can update our fill/stroke/etc. on our own
        this.defs = null;

        if ( usesFill ) {
          if ( !this.fillState ) {
            this.fillState = new Fillable.FillSVGState();
          } else {
            this.fillState.initialize();
          }
        }

        if ( usesStroke ) {
          if ( !this.strokeState ) {
            this.strokeState = new Strokable.StrokeSVGState();
          } else {
            this.strokeState.initialize();
          }
        }

        return this; // allow for chaining
      },

      // to be used by our passed in options.updateSVG
      updateFillStrokeStyle: function( element ) {
        if ( usesFill && this.dirtyFill ) {
          this.fillState.updateFill( this.defs, this.node._fill );
        }
        var strokeParameterDirty;
        if ( usesStroke ) {
          if ( this.dirtyStroke ) {
            this.strokeState.updateStroke( this.defs, this.node._stroke );
          }
          strokeParameterDirty = this.dirtyLineWidth || this.dirtyLineOptions;
          if ( strokeParameterDirty ) {
            this.strokeState.updateStrokeParameters( this.node );
          }
        }
        if ( ( usesFill && this.dirtyFill ) || ( usesStroke && ( this.dirtyStroke || strokeParameterDirty ) ) ) {
          element.setAttribute( 'style', ( usesFill ? this.fillState.style : '' ) + ( usesStroke ? this.strokeState.baseStyle + this.strokeState.extraStyle : '' ) );
        }
      },

      updateSVG: function() {
        if ( this.paintDirty ) {
          updateSVGSelf.call( this, this.node, this.svgElement );
        }

        // clear all of the dirty flags
        this.setToClean();
      },

      updateDefs: function( defs ) {
        this.defs = defs;

        updateDefsSelf && updateDefsSelf.call( this, defs );

        usesFill && this.fillState.updateDefs( defs );
        usesStroke && this.strokeState.updateDefs( defs );
      },

      onAttach: function( node ) {

      },

      // release the SVG elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
      onDetach: function( node ) {
        //OHTWO TODO: are we missing the disposal?
        if ( !keepElements ) {
          // clear the references
          this.svgElement = null;
        }

        // release any defs, and dispose composed state objects
        updateDefsSelf && updateDefsSelf.call( this, null );
        usesFill && this.fillState.dispose();
        usesStroke && this.strokeState.dispose();

        this.defs = null;
      },

      setToClean: function() {
        this.setToCleanState();
      }
    } );

    // mix-in
    stateType( type );

    // set up pooling
    /* jshint -W064 */
    SelfDrawable.Poolable( type );

    return type;
  };

  return SVGSelfDrawable;
} );

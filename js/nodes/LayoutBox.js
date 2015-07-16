// Copyright 2002-2014, University of Colorado Boulder

/**
 * LayoutBox lays out its children in a row, either horizontally or vertically (based on an optional parameter).
 * VBox and HBox are convenience subtypes that specify the orientation.
 * See https://github.com/phetsims/scenery/issues/281
 *
 * @author Sam Reid
 * @author Aaron Davis
 * @author Chris Malley (PixelZoom, Inc.)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  var defaultSpacing = function( child, nextChild ) { return 0; };

  /**
   * Promotes spacing to a function.
   * @param {number|function} spacing
   * @returns {function}
   */
  var spacingValueToFunction = function( spacing ) {
    assert && assert( typeof spacing === 'number' || typeof spacing === 'function', 'unsupport type for spacing' );
    if ( typeof spacing === 'number' ) {
      return function( child, nextChild ) { return spacing; };
    }
    else {
      return spacing;
    }
  };

  /**
   * @param {Object} [options] Same as Node.constructor.options with the following additions:
   * @constructor
   */
  scenery.LayoutBox = function LayoutBox( options ) {

    options = _.extend( {

      //The default orientation, chosen by popular vote.  At the moment there are around 436 VBox references and 338 HBox references
      orientation: 'vertical',

      // The spacing can be a number or a function.  If a number, then it will be the spacing between each object.
      // If a function, then the function will have the signature function( child, nextChild ){...} which returns
      // the spacing between the current and next child. For the last child, nextChild will be undefined.
      spacing: defaultSpacing,

      //How to line up the items
      align: 'center',

      //By default, update the layout when children are added/removed/resized, see #116
      resize: true
    }, options ); // @private

    // validate options
    assert && assert( options.orientation === 'vertical' || options.orientation === 'horizontal' );
    if ( options.orientation === 'vertical' ) {
      assert && assert( options.align === 'center' || options.align === 'left' || options.align === 'right', 'illegal alignment: ' + options.align );
    }
    else {
      assert && assert( options.align === 'center' || options.align === 'top' || options.align === 'bottom', 'illegal alignment: ' + options.align );
    }

    this.orientation = options.orientation; // @private
    this.align = options.align; // @private
    this.resize = options.resize; // @private
    this._spacing = spacingValueToFunction( options.spacing ); // @private {function}

    Node.call( this );

    this.boundsListener = this.updateLayout.bind( this ); // @private
    this.updatingLayout = false; // @private flag used to short-circuit updateLayout and prevent stackoverflow

    // Apply the supplied options, including children.
    // The layout calls are triggered if (a) options.resize is set to true or (b) during initialization
    // When true, the this.inited flag signifies that the initial layout is being done.
    this.inited = false; // @private
    this.mutate( options );
    this.inited = true;
  };
  var LayoutBox = scenery.LayoutBox;

  return inherit( Node, LayoutBox, {

    // Lay out the child components on startup, or when the children sizes change or when requested by a call to updateLayout
    // @private, do not call directly, use updateLayout
    layout: function() {

      var children = this.getChildren(); // call this once, since it returns a copy
      var i = 0;
      var child;

      //Logic for layout out the components.
      //Aaron and Sam looked at factoring this out, but the result looked less readable since each attribute
      //would have to be abstracted over.
      if ( this.orientation === 'vertical' ) {
        var minX = _.min( _.map( children, function( child ) {return child.left;} ) );
        var maxX = _.max( _.map( children, function( child ) {return child.left + child.width;} ) );
        var centerX = (maxX + minX) / 2;

        // Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, 
        // instead just set it with the {y:number} option.
        var y = 0;
        for ( i = 0; i < children.length; i++ ) {
          child = children[ i ];
          child.top = y;

          //Set the position horizontally
          if ( this.align === 'left' ) {
            child.left = minX;
          }
          else if ( this.align === 'right' ) {
            child.right = maxX;
          }
          else { // 'center'
            child.centerX = centerX;
          }

          //Move to the next vertical position.
          y += child.height + this._spacing( child, children[ i + 1 ] );
        }
      }
      else {
        var minY = _.min( _.map( children, function( child ) {return child.top;} ) );
        var maxY = _.max( _.map( children, function( child ) {return child.top + child.height;} ) );
        var centerY = (maxY + minY) / 2;

        //Start at x=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {x:number} option.
        var x = 0;
        for ( i = 0; i < children.length; i++ ) {
          child = children[ i ];
          child.left = x;

          //Set the position horizontally
          if ( this.align === 'top' ) {
            child.top = minY;
          }
          else if ( this.align === 'bottom' ) {
            child.bottom = maxY;
          }
          else { // 'center'
            child.centerY = centerY;
          }

          //Move to the next horizontal position.
          x += child.width + this._spacing( child, children[ i + 1 ] );
        }
      }
    },

    // Update the layout of this LayoutBox. Called automatically during initialization, when children change (if resize is true)
    // or when client wants to call this public method for any reason.
    updateLayout: function() {
      //Bounds of children are changed in updateLayout, we don't want to stackoverflow, so bail if already updating layout
      if ( !this.updatingLayout ) {
        this.updatingLayout = true;
        this.layout();
        this.updatingLayout = false;
      }
    },

    //Override the child mutators to updateLayout
    //Have to listen to the child bounds individually because there are a number of possible ways to change the child
    //bounds without changing the overall bounds.
    // @override
    insertChild: function( index, node ) {
      //Support up to two args for overrides

      //Remove event listeners from any nodes (will be added back later if the node was not removed)
      var layoutBox = this;
      if ( this.resize ) {
        this.getChildren().forEach( function( child ) {
          if ( child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.removeEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }

      //Super call
      Node.prototype.insertChild.call( this, index, node );

      // Update the layout (a) if it should be dynamic or (b) during initialization
      if ( this.resize || !this.inited ) {
        this.updateLayout();
      }

      //Add event listeners for any current children (if it should be dynamic)
      if ( this.resize ) {
        this.getChildren().forEach( function( child ) {
          if ( !child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.addEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }
    },

    //Overrides the version in Node to listen for bounds changes
    // @override
    removeChildWithIndex: function( node, indexOfChild ) {

      //Remove event listeners from any nodes (will be added back later if the node was not removed)
      var layoutBox = this;
      if ( this.resize ) {
        this.getChildren().forEach( function( child ) {
          if ( child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.removeEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }

      //Super call
      Node.prototype.removeChildWithIndex.call( this, node, indexOfChild );

      // Update the layout (a) if it should be dynamic or (b) during initialization
      if ( this.resize || !this.inited ) {
        this.updateLayout();
      }

      //Add event listeners for any current children (if it should be dynamic)
      if ( this.resize ) {
        this.getChildren().forEach( function( child ) {
          if ( !child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.addEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }
    },

    /**
     * Sets spacing between items in the box.
     * @param {number|function} spacing
     */
    setSpacing: function( spacing ) {
      if ( this._spacing !== spacing ) {
        this._spacing = spacingValueToFunction( spacing );
        this.updateLayout();
      }
    },
    set spacing( value ) { this.setSpacing( value ); },

    /**
     * Gets the function used to set the spacing between items in the box.
     * @returns {function}
     */
    getSpacing: function() { return this._spacing; },
    get spacing() { return this.getSpacing(); }
  } );
} );
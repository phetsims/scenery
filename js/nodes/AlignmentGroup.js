// Copyright 2016-2016, University of Colorado Boulder

/**
 * A group of containers that follow the constraints:
 * 1. Every container will have the same bounds, with an upper-left of (0,0)
 * 2. The container sizes will be the smallest possible to fit every container's content (with respective padding).
 * 3. Each container is responsible for positioning its content in its bounds (with customizable alignment and padding).
 *
 * Containers can be dynamically created and disposed, and only active containers will be considered for the bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  /**
   * Creates an alignment group that can be composed of multiple containers (made with createContainer()).
   * @constructor
   * @public
   */
  function AlignmentGroup() {
    // @private {Array.<AlignmentContainer>}
    this._containers = [];

    // @private {boolean} - Gets locked when certain layout is performed.
    this._resizeLock = false;
  }

  scenery.register( 'AlignmentGroup', AlignmentGroup );

  inherit( Object, AlignmentGroup, {
    /**
     * Creates a container with the given content and options.
     * @public
     *
     * @param {Node} content - Note that the content may be repositioned into place.
     * @param {Object} [options] - See AlignmentContainer's constructor below for specific alignment options.
     *                             Will also be passed to the node's constructor.
     */
    createContainer: function( content, options ) {
      assert && assert( content instanceof Node );

      var container = new AlignmentContainer( content, this, options );
      this._containers.push( container );

      // Trigger an update when a container is added
      this.updateLayout();

      return container;
    },

    /**
     * Updates the localBounds and alignment for each container.
     * @private
     */
    updateLayout: function() {
      if ( this._resizeLock ) { return; }
      this._resizeLock = true;

      // Compute the maximum dimension of our containers' content
      var maxWidth = 0;
      var maxHeight = 0;
      for ( var i = 0; i < this._containers.length; i++ ) {
        var container = this._containers[ i ];

        var bounds = container.getContentBounds();

        // Ignore bad bounds
        if ( bounds.isEmpty() || !bounds.isFinite() ) {
          continue;
        }

        maxWidth = Math.max( maxWidth, bounds.width );
        maxHeight = Math.max( maxHeight, bounds.height );
      }

      if ( maxWidth > 0 && maxHeight > 0 ) {
        // Apply that maximum dimension for each container
        for ( i = 0; i < this._containers.length; i++ ) {
          this._containers[ i ].updateLayout( maxWidth, maxHeight );
        }
      }

      this._resizeLock = false;
    },

    /**
     * Lets the group know that the container has had its content resized.
     * @private
     *
     * @param {AlignmentContainer}
     */
    onContainerContentResized: function( container ) {
      // TODO: in the future, we could only update this specific container if the others don't need updating.
      this.updateLayout();
    },

    /**
     * Handles disposal of the container
     * @private
     */
    disposeContainer: function( container ) {
      arrayRemove( this._containers, container );

      // Trigger an update when a container is removed
      this.updateLayout();
    },

    /**
     * Dispose all of the containers.
     * @public
     */
    dispose: function() {
      for ( var i = this._containers.length - 1; i >= 0; i-- ) {
        this._containers[ i ].dispose();
      }
    }
  } );

  /**
   * @constructor
   * @private
   *
   * @param {Node} content
   * @param {AlignmentGroup} alignmentGroup
   * @param {Object} [options] - See the _.extend below in AlignmentContainer for documentation. Also passed to Node.
   */
  function AlignmentContainer( content, alignmentGroup, options ) {
    options = _.extend( {
      xAlign: 'center', // {string} - 'left', 'center', or 'right', for horizontal positioning
      yAlign: 'center', // {string} - 'top', 'center', or 'bottom', for vertical positioning
      xMargin: 0, // {number} - Non-negative margin that should exist on the left and right of the content
      yMargin: 0 // {number} - Non-negative margin that should exist on the top and bottom of the content.
    }, options );

    assert && assert( options.xAlign === 'left' || options.xAlign === 'center' || options.xAlign === 'right',
      'xAlign should be one of: \'left\', \'center\', or \'right\'' );
    assert && assert( options.yAlign === 'left' || options.yAlign === 'center' || options.yAlign === 'right',
      'yAlign should be one of: \'top\', \'center\', or \'bottom\'' );
    assert && assert( typeof options.xMargin === 'number' && isFinite( options.xMargin ) && options.xMargin >= 0,
      'xMargin should be a finite non-negative number' );
    assert && assert( typeof options.yMargin === 'number' && isFinite( options.yMargin ) && options.yMargin >= 0,
      'yMargin should be a finite non-negative number' );

    // @private {string}
    this._xAlign = options.xAlign;
    this._yAlign = options.yAlign;

    // @private {number}
    this._xMargin = options.xMargin;
    this._yMargin = options.yMargin;

    // @private {Node}
    this._content = content;

    // @private {AlignmentGroup}
    this._alignmentGroup = alignmentGroup;

    // @private {function} - Callback for when bounds change
    this._contentBoundsListener = alignmentGroup.onContainerContentResized.bind( alignmentGroup, this );

    this._content.on( 'bounds', this._contentBoundsListener );

    Node.call( this, _.extend( options, {
      children: [ this._content ]
    } ) );
  }

  inherit( Node, AlignmentContainer, {
    /**
     * Returns the bounding box of this container's content. This will include any margins.
     * @private
     *
     * @returns {Bounds2}
     */
    getContentBounds: function() {
      return this._content.bounds.dilatedXY( this._xMargin, this._yMargin );
    },

    /**
     * Updates the layout of this alignment container.
     * @private
     *
     * @param {number} width
     * @param {number} height
     */
    updateLayout: function( width, height ) {
      this.localBounds = new Bounds2( 0, 0, width, height );

      if ( this._xAlign === 'center' ) {
        this._content.centerX = this.localBounds.centerX;
      }
      else if ( this._xAlign === 'left' ) {
        this._content.left = this.localBounds.left + this._xMargin;
      }
      else if ( this._xAlign === 'right' ) {
        this._content.right = this.localBounds.right - this._xMargin;
      }
      else {
        assert && assert( 'Bad xAlign: ' + this._xAlign );
      }

      if ( this._yAlign === 'center' ) {
        this._content.centerY = this.localBounds.centerY;
      }
      else if ( this._yAlign === 'top' ) {
        this._content.top = this.localBounds.top + this._yMargin;
      }
      else if ( this._yAlign === 'bottom' ) {
        this._content.bottom = this.localBounds.bottom - this._yMargin;
      }
      else {
        assert && assert( 'Bad yAlign: ' + this._yAlign );
      }

      assert && assert( this.localBounds.containsBounds( this._content.bounds ),
        'All of our contents should be contained in our localBounds' );
    },

    /**
     * Disposes this container, so that the alignment group won't update this container, and won't use its bounds for
     * laying out the other containers.
     * @public
     */
    dispose: function() {
      this._content.off( 'bounds', this._contentBoundsListener );

      this._alignmentGroup.disposeContainer( this );
    }
  } );

  return AlignmentGroup;
} );

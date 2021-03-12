// Copyright 2013-2020, University of Colorado Boulder

/**
 * The PointerOverlay shows pointer locations in the scene.  This is useful when recording a session for interviews or when a teacher is broadcasting
 * a tablet session on an overhead projector.  See https://github.com/phetsims/scenery/issues/111
 *
 * Each pointer is rendered in a different <svg> so that CSS3 transforms can be used to make performance smooth on iPad.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import PDOMPointer from '../input/PDOMPointer.js';
import scenery from '../scenery.js';
import svgns from '../util/svgns.js';
import Utils from '../util/Utils.js';

class PointerOverlay {
  /**
   * @param {Display} display
   * @param {Node} rootNode - the root node of our display
   */
  constructor( display, rootNode ) {
    this.display = display;
    this.rootNode = rootNode;

    // add element to show the pointers
    this.pointerSVGContainer = document.createElement( 'div' );
    this.pointerSVGContainer.style.position = 'absolute';
    this.pointerSVGContainer.style.top = 0;
    this.pointerSVGContainer.style.left = 0;
    this.pointerSVGContainer.style[ 'pointer-events' ] = 'none';

    const innerRadius = 10;
    const strokeWidth = 1;
    const diameter = ( innerRadius + strokeWidth / 2 ) * 2;
    const radius = diameter / 2;

    // Resize the parent div when the rootNode is resized
    display.sizeProperty.lazyLink( dimension => {
      this.pointerSVGContainer.setAttribute( 'width', dimension.width );
      this.pointerSVGContainer.setAttribute( 'height', dimension.height );
      this.pointerSVGContainer.style.clip = `rect(0px,${dimension.width}px,${dimension.height}px,0px)`;
    } );

    const scratchMatrix = Matrix3.IDENTITY.copy();

    //Display a pointer that was added.  Use a separate SVG layer for each pointer so it can be hardware accelerated, otherwise it is too slow just setting svg internal attributes
    this.pointerAdded = pointer => {

      const svg = document.createElementNS( svgns, 'svg' );
      svg.style.position = 'absolute';
      svg.style.top = 0;
      svg.style.left = 0;
      svg.style[ 'pointer-events' ] = 'none';

      Utils.prepareForTransform( svg );

      //Fit the size to the display
      svg.setAttribute( 'width', diameter );
      svg.setAttribute( 'height', diameter );

      const circle = document.createElementNS( svgns, 'circle' );

      //use css transform for performance?
      circle.setAttribute( 'cx', innerRadius + strokeWidth / 2 );
      circle.setAttribute( 'cy', innerRadius + strokeWidth / 2 );
      circle.setAttribute( 'r', innerRadius );
      circle.setAttribute( 'style', 'fill:black;' );
      circle.setAttribute( 'style', 'stroke:white;' );
      circle.setAttribute( 'opacity', '0.4' );

      const updateToPoint = point => Utils.applyPreparedTransform( scratchMatrix.setToTranslation( point.x - radius, point.y - radius ), svg );

      //Add a move listener to the pointer to update position when it has moved
      const pointerRemoved = () => {

        // For touche-like events that get a touch up event, remove them.  But when the mouse button is released, don't stop
        // showing the mouse location
        if ( pointer.isTouchLike() ) {
          this.pointerSVGContainer.removeChild( svg );
          pointer.removeInputListener( moveListener );
        }
      };
      const moveListener = {

        // Mouse/Touch/Pen
        move: () => {
          pointer.point && updateToPoint( pointer.point );
        },
        up: pointerRemoved,
        cancel: pointerRemoved,

        // PDOMPointer
        focus: () => {
          if ( pointer instanceof PDOMPointer && pointer.point ) {
            updateToPoint( pointer.point );
            this.pointerSVGContainer.appendChild( svg );
          }
        },
        blur: () => {
          this.pointerSVGContainer.contains( svg ) && this.pointerSVGContainer.removeChild( svg );
        }
      };
      pointer.addInputListener( moveListener );

      moveListener.move();
      svg.appendChild( circle );
      this.pointerSVGContainer.appendChild( svg );
    };
    display._input.addPointerAddedListener( this.pointerAdded );

    //if there is already a mouse, add it here
    // TODO: if there already other non-mouse touches, could be added here
    if ( display._input && display._input.mouse ) {
      this.pointerAdded( display._input.mouse );
    }

    this.domElement = this.pointerSVGContainer;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    this.display._input.removePointerAddedListener( this.pointerAdded );
  }

  /**
   * @public
   */
  update() {
  }
}

scenery.register( 'PointerOverlay', PointerOverlay );
export default PointerOverlay;
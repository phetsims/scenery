// Copyright 2021, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { SceneryEvent } from '../imports.js';

type SceneryListenerFunction = ( event: SceneryEvent ) => void;

interface IInputListener {
  interrupt?: () => void;
  cursor?: string | null;

  focus?: SceneryListenerFunction;
  blur?: SceneryListenerFunction;
  focusin?: SceneryListenerFunction;
  focusout?: SceneryListenerFunction;

  keydown?: SceneryListenerFunction;
  keyup?: SceneryListenerFunction;

  click?: SceneryListenerFunction;
  input?: SceneryListenerFunction;
  change?: SceneryListenerFunction;

  down?: SceneryListenerFunction;
  mousedown?: SceneryListenerFunction;
  touchdown?: SceneryListenerFunction;
  pendown?: SceneryListenerFunction;

  up?: SceneryListenerFunction;
  mouseup?: SceneryListenerFunction;
  touchup?: SceneryListenerFunction;
  penup?: SceneryListenerFunction;

  cancel?: SceneryListenerFunction;
  mousecancel?: SceneryListenerFunction;
  touchcancel?: SceneryListenerFunction;
  pencancel?: SceneryListenerFunction;

  move?: SceneryListenerFunction;
  mousemove?: SceneryListenerFunction;
  touchmove?: SceneryListenerFunction;
  penmove?: SceneryListenerFunction;

  wheel?: SceneryListenerFunction;
  mousewheel?: SceneryListenerFunction;
  touchwheel?: SceneryListenerFunction;
  penwheel?: SceneryListenerFunction;

  enter?: SceneryListenerFunction;
  mouseenter?: SceneryListenerFunction;
  touchenter?: SceneryListenerFunction;
  penenter?: SceneryListenerFunction;

  exit?: SceneryListenerFunction;
  mouseexit?: SceneryListenerFunction;
  touchexit?: SceneryListenerFunction;
  penexit?: SceneryListenerFunction;

  over?: SceneryListenerFunction;
  mouseover?: SceneryListenerFunction;
  touchover?: SceneryListenerFunction;
  penover?: SceneryListenerFunction;

  out?: SceneryListenerFunction;
  mouseout?: SceneryListenerFunction;
  touchout?: SceneryListenerFunction;
  penout?: SceneryListenerFunction;
}

export default IInputListener;
export type { SceneryListenerFunction };
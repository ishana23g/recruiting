import { Form, FormField, FormLabel } from '@radix-ui/react-form';
import { Button, Card, Flex, Heading, Separator, TextField } from '@radix-ui/themes';
import _ from 'lodash';
import React, { useCallback, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Routes } from 'routes';

type FormValue = number | '';
interface FormData {
  Body1: {
    position: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    velocity: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    mass: FormValue;
  };
  Body2: {
    position: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    velocity: {
      x: FormValue;
      y: FormValue;
      z: FormValue;
    }
    mass: FormValue;
  };
}

const SimulateForm: React.FC = () => {
  const navigate = useNavigate();

  const [n, setN] = useState<number>(50);
  const [iterations, setIterations] = useState<number>(500);
  const [seed, setSeed] = useState<string>('42');
  const [heavyCenter, setHeavyCenter] = useState<boolean>(true);

  const [formData, setFormData] = useState<FormData>({
    Body1: { position: {x: -0.73, y: 0, z: 0}, velocity: {x: 0, y: -0.0015, z: 0}, mass: 1 },
    Body2: { position: {x: 60.34, y: 0, z: 0}, velocity: {x: 0, y: 0.13, z: 0}, mass: 0.0123 },
  });

  // const runRandom = useCallback(async () => {
  //   try {
  //     const params = new URLSearchParams({
  //       n: String(n),
  //       iterations: String(iterations),
  //     });
  //     if (seed !== '') params.set('seed', seed);

  //     // Trigger the random sim (GET /simulation handles n, iterations, seed)
  //     const response = await fetch(`http://localhost:8000/simulation?${params.toString()}`, {
  //       method: 'GET',
  //     });
  //     if (!response.ok) {
  //       throw new Error('Network response was not ok');
  //     }

  //     // Go to results page (same behavior as handleSubmit)
  //     navigate(Routes.SIMULATION);
  //   } catch (error) {
  //     console.error('Error:', error);
  //   }
  // }, [n, iterations, seed, navigate]);

  const runRandomWithWorkers = useCallback(async (workers: '1' | 'auto') => {
    try {
      const params = new URLSearchParams({
        n: String(n),
        iterations: String(iterations),
        workers,
      });
      if (seed !== '') params.set('seed', seed);
      if (heavyCenter) params.set('heavy_center', 'true');
  
      const response = await fetch(`http://localhost:8000/simulation/random?${params.toString()}`, {
        method: 'GET',
      });
      if (!response.ok) throw new Error('Network response was not ok');
      navigate(Routes.SIMULATION);
    } catch (error) {
      console.error('Error:', error);
    }
  }, [n, iterations, seed, heavyCenter, navigate]);
    
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    let newValue: FormValue = value === '' ? '' : parseFloat(value);
    setFormData((prev) => _.set({ ...prev }, name, newValue));
  }, []);

  // const handleSubmit = useCallback(
  //   async (e: React.FormEvent) => {
  //     e.preventDefault();
  //     try {
  //       const response = await fetch('http://localhost:8000/simulation', {
  //         method: 'POST',
  //         headers: {
  //           'Content-Type': 'application/json',
  //         },
  //         body: JSON.stringify(formData),
  //       });
  //       if (!response.ok) {
  //         throw new Error('Network response was not ok');
  //       }
  //       navigate(Routes.SIMULATION);
  //     } catch (error) {
  //       console.error('Error:', error);
  //     }
  //   },
  //   [formData]
  // );

  const submitRegular = useCallback(async (workers: '1' | 'auto') => {
    try {
      const url = `http://localhost:8000/simulation?workers=${workers}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error('Network response was not ok');
      navigate(Routes.SIMULATION);
    } catch (error) {
      console.error('Error:', error);
    }
  }, [formData, navigate]);
  
  return (
    <div
      style={{
        position: 'absolute',
        top: '5%',
        left: 'calc(50% - 200px)',
        overflow: 'scroll',
      }}
    >
      {/* Card: https://www.radix-ui.com/themes/docs/components/card */}
      <Card
        style={{
          width: '400px',
        }}
      >
        <Heading as="h2" size="4" weight="bold" mb="4">
          Run a Simulation
        </Heading>
        <Link to={Routes.SIMULATION}>View previous simulation</Link>
        <Separator size="4" my="5" />
        {/* Quick random N-body section */}
        <Heading as="h3" size="3" weight="bold">Random N-bodies</Heading>
        <Form>
          <FormField name="random.n">
            <FormLabel htmlFor="random.n">N-bodies</FormLabel>
            <TextField.Root
              type="number"
              id="random.n"
              value={n}
              onChange={(e: { target: { value: any; }; }) => setN(parseInt(e.target.value || '0', 10))}
              placeholder="N (bodies)"
            />
          </FormField>
          <FormField name="random.iterations">
            <FormLabel htmlFor="random.iterations">Time steps</FormLabel>
            <TextField.Root
              type="number"
              id="random.iterations"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value || '0', 10))}
              placeholder="Iterations"
            />
          </FormField>
          <FormField name="random.seed">
            <FormLabel htmlFor="random.seed">Seed (optional)</FormLabel>
            <TextField.Root
              type="number"
              id="random.seed"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="e.g. 42"
            />
          </FormField>
          <FormField name="random.heavyCenter">
          <FormLabel htmlFor="random.heavyCenter">Central massive body</FormLabel>
          <input
            id="random.heavyCenter"
            type="checkbox"
            checked={heavyCenter}
            onChange={(e) => setHeavyCenter(e.target.checked)}
            style={{ marginTop: 8 }}
          />
        </FormField>
          <Flex justify="center" gap="3" m="3" wrap="wrap">
            <Button type="button" onClick={() => runRandomWithWorkers('1')}>Run random (single-thread)</Button>
            <Button type="button" onClick={() => runRandomWithWorkers('auto')}>Run random (parallel)</Button>
          </Flex>
        </Form>
        <Separator size="4" my="5" />
        <Form>
          {/* 
            *********************************
            Body1
            *********************************
            */}
          <Heading as="h3" size="3" weight="bold">
            Body1
          </Heading>
          {/* Form: https://www.radix-ui.com/primitives/docs/components/form */}
          <FormField name="Body1.position.x">
            <FormLabel htmlFor="Body1.position.x">Initial X-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.position.x"
              name="Body1.position.x"
              value={formData.Body1.position.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.position.y">
            <FormLabel htmlFor="Body1.position.y">Initial Y-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.position.y"
              name="Body1.position.y"
              value={formData.Body1.position.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.position.z">
            <FormLabel htmlFor="Body1.position.z">Initial Z-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.position.z"
              name="Body1.position.z"
              value={formData.Body1.position.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.velocity.x">
            <FormLabel htmlFor="Body1.velocity.x">Initial X-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.velocity.x"
              name="Body1.velocity.x"
              value={formData.Body1.velocity.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.velocity.y">
            <FormLabel htmlFor="Body1.velocity.y">Initial Y-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.velocity.y"
              name="Body1.velocity.y"
              value={formData.Body1.velocity.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.velocity.z">
            <FormLabel htmlFor="Body1.velocity.z">Initial Z-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.velocity.z"
              name="Body1.velocity.z"
              value={formData.Body1.velocity.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body1.mass">
            <FormLabel htmlFor="Body1.mass">Mass</FormLabel>
            <TextField.Root
              type="number"
              id="Body1.mass"
              name="Body1.mass"
              value={formData.Body1.mass}
              onChange={handleChange}
              required
            />
          </FormField>
          {/* 
            *********************************
            Body2
            *********************************
             */}
          <Heading as="h3" size="3" weight="bold" mt="4">
            Body2
          </Heading>
          <FormField name="Body2.position.x">
            <FormLabel htmlFor="Body2.position.x">Initial X-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.position.x"
              name="Body2.position.x"
              value={formData.Body2.position.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.position.y">
            <FormLabel htmlFor="Body2.position.y">Initial Y-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.position.y"
              name="Body2.position.y"
              value={formData.Body2.position.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.position.z">
            <FormLabel htmlFor="Body2.position.z">Initial Z-position</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.position.z"
              name="Body2.position.z"
              value={formData.Body2.position.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.velocity.x">
            <FormLabel htmlFor="Body2.velocity.x">Initial X-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.velocity.x"
              name="Body2.velocity.x"
              value={formData.Body2.velocity.x}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.velocity.y">
            <FormLabel htmlFor="Body2.velocity.y">Initial Y-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.velocity.y"
              name="Body2.velocity.y"
              value={formData.Body2.velocity.y}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.velocity.z">
            <FormLabel htmlFor="Body2.velocity.z">Initial Z-velocity</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.velocity.z"
              name="Body2.velocity.z"
              value={formData.Body2.velocity.z}
              onChange={handleChange}
              required
            />
          </FormField>
          <FormField name="Body2.mass">
            <FormLabel htmlFor="Body2.mass">Mass</FormLabel>
            <TextField.Root
              type="number"
              id="Body2.mass"
              name="Body2.mass"
              value={formData.Body2.mass}
              onChange={handleChange}
              required
            />
          </FormField>
          <Flex justify="center" m="5">
          <Button type="button" onClick={() => submitRegular('1')} mr="2">Submit (single-thread)</Button>
          <Button type="button" onClick={() => submitRegular('auto')}>Submit (parallel)</Button>
          </Flex>
        </Form>
      </Card>
    </div>
  );
};

export default SimulateForm;
